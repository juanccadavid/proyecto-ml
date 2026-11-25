-- Migración 004: contrato v2.2.0, los códigos del extracto real (ver docs/datos_reales.md).
-- Para una base ya creada con supabase/asociados.sql v2.0.0 o v2.1.0 (44 columnas). Pegar en Supabase > SQL
-- Editor > Run. Una base nueva lo trae en supabase/asociados.sql (make contrato-sql).
--
-- Si la tabla es de un contrato v1 (74 columnas, ESTRATO numérico), esta migración se detiene sin cambiar nada:
-- de v1 a v2 cambió casi todo el esquema y no hay migración. La tabla solo tiene datos de prueba, así que se
-- recrea: en el SQL Editor, "drop table public.asociados;" y luego pegar y correr supabase/asociados.sql (la tabla
-- public.equipo, con los correos del equipo, no se toca).
--
-- - Códigos del diccionario de datos en SEXO, ESTUDIOS, ESTADOCIVIL, TIPOCONTRATO, ESTRATO y
--   #SOLICITUD_DISMINUCIONCUOTAS (SI/NO).
-- - ESTRATO es opcional: el extracto real trae un tercio en blanco.
-- - FECHAINGRESOLABORA ya no exige 1950 en adelante ni ser posterior al nacimiento: son advertencias del
--   contrato, que no bloquean (la fecha laboral la reporta la empresa y no entra al modelo).
--
-- Las restricciones nuevas van con "not valid": se exigen a las filas nuevas sin revisar las que ya están (datos
-- de prueba con los códigos anteriores, p. ej. ESTUDIOS = 'BACHILLER'). Para exigirlas también a esas, borrarlas
-- o corregirlas y correr: alter table public.asociados validate constraint <nombre>;

begin;  -- todo o nada: si algo falla, la tabla queda como estaba

do $$
begin
  if not exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'asociados' and column_name = 'estrato' and data_type = 'text'
  ) then
    raise exception 'public.asociados no es del contrato v2 (ESTRATO no es texto). Recrea la tabla: drop table public.asociados; y corre supabase/asociados.sql';
  end if;
end $$;

alter table public.asociados alter column estrato drop not null;

alter table public.asociados drop constraint if exists asociados_fechaingresolabora_check;
alter table public.asociados drop constraint if exists ingreso_laboral_posterior_nacimiento;

alter table public.asociados drop constraint if exists asociados_sexo_check;
alter table public.asociados add constraint asociados_sexo_check
  check (sexo <> '' and sexo in ('F', 'M')) not valid;

alter table public.asociados drop constraint if exists asociados_estrato_check;
alter table public.asociados add constraint asociados_estrato_check
  check (estrato <> '' and estrato in ('1', '2', '3', '4', '5', '6')) not valid;

alter table public.asociados drop constraint if exists asociados_estudios_check;
alter table public.asociados add constraint asociados_estudios_check
  check (estudios <> '' and estudios in ('O', 'U', 'G', 'S', 'E', 'P', 'T')) not valid;

alter table public.asociados drop constraint if exists asociados_estadocivil_check;
alter table public.asociados add constraint asociados_estadocivil_check
  check (estadocivil <> '' and estadocivil in ('C', 'T', 'V', 'S', 'D', 'U', 'F', 'P')) not valid;

alter table public.asociados drop constraint if exists asociados_tipocontrato_check;
alter table public.asociados add constraint asociados_tipocontrato_check
  check (tipocontrato <> '' and tipocontrato in ('2', '1', '15', 'O', '0')) not valid;

alter table public.asociados drop constraint if exists asociados_num_solicitud_disminucioncuotas_check;
alter table public.asociados add constraint asociados_num_solicitud_disminucioncuotas_check
  check (num_solicitud_disminucioncuotas <> '' and num_solicitud_disminucioncuotas in ('SI', 'NO')) not valid;

comment on table public.asociados is 'Contrato asociados v2.2.0';
comment on column public.asociados.estrato is 'ESTRATO — Opcional; el extracto real trae vacíos';

commit;
