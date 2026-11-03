-- Migración 002: formulario solo para el equipo, en una tabla de entrada (contrato v1.1.0).
-- Para una base ya creada con supabase/asociados.sql anterior. Pegar en Supabase > SQL Editor > Run.
-- Generado con contrato.migracion_formulario(); una base nueva lo trae en supabase/asociados.sql.
-- Después: agregar los correos del equipo (Table Editor > equipo, o en SQL):
--   insert into public.equipo (email, nombre) values ('correo@dominio.com', 'Nombre');

alter table public.asociados enable row level security;

-- Formulario: solo el equipo puede insertar, y en una tabla de entrada que se revisa.
-- 1) La clave pública (anon) ya no inserta en public.asociados.
drop policy if exists anon_solo_inserta on public.asociados;
revoke insert on public.asociados from anon;
revoke usage on sequence public.asociados_id_seq from anon;

-- 2) El equipo: correos que pueden enviar el formulario. Se administra en el panel de Supabase
--    (Table Editor) o con la clave secreta; con RLS y sin políticas, anon y authenticated no la ven.
create table if not exists public.equipo (
  email text primary key check (email = lower(email)),
  nombre text,
  agregado_en timestamptz not null default now()
);
alter table public.equipo enable row level security;

create or replace function public.es_del_equipo() returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (select 1 from public.equipo where email = lower(coalesce(auth.jwt() ->> 'email', '')))
$$;
revoke all on function public.es_del_equipo() from public;
grant execute on function public.es_del_equipo() to authenticated;

-- 3) Tabla de entrada: mismas columnas y reglas que public.asociados, sin UNIQUE. make supabase-revisar pasa lo
--    válido a public.asociados y marca cada fila (revision) como aprobada o rechazada, con motivo_revision.
create table if not exists public.asociados_entrada (
  id bigint generated always as identity primary key,
  creado_en timestamptz not null default now(),
  enviado_por text not null default lower(coalesce(auth.jwt() ->> 'email', '')),
  revision text not null default 'pendiente' check (revision in ('pendiente', 'aprobada', 'rechazada')),
  revisado_en timestamptz,
  motivo_revision text,
  cedula bigint not null check (cedula >= 1 and cedula <= 9999999999),
  primerapellido text not null check (primerapellido <> ''),
  segundoapellido text check (segundoapellido <> ''),
  nombres text not null check (nombres <> ''),
  segundonombre text check (segundonombre <> ''),
  nombreintegrado text not null check (nombreintegrado <> ''),
  direccion text not null check (direccion <> ''),
  tipoidentificacion text not null check (tipoidentificacion <> '' and tipoidentificacion in ('CC', 'CE', 'TI', 'PA')),
  relacion text not null check (relacion <> '' and relacion in ('ASOCIADO')),
  agencia bigint not null check (agencia >= 1 and agencia <= 99),
  telefono1 text not null check (telefono1 <> '' and telefono1 ~ '^[0-9]{7,10}$'),
  extension1 text check (extension1 <> '' and extension1 ~ '^[0-9]{1,6}$'),
  telefono2 text check (telefono2 <> '' and telefono2 ~ '^[0-9]{7,10}$'),
  extension2 text check (extension2 <> '' and extension2 ~ '^[0-9]{1,6}$'),
  apartadoaereo text check (apartadoaereo <> ''),
  sexo text not null check (sexo <> '' and sexo in ('F', 'M')),
  codpaiscedula bigint not null check (codpaiscedula >= 1 and codpaiscedula <= 999),
  codciudadcedula bigint not null check (codciudadcedula >= 1 and codciudadcedula <= 99999),
  email text not null check (email <> '' and email ~ '^[^@ ]+@[^@ ]+[.][^@ ]+$'),
  fecharetiro date check (fecharetiro >= date '1990-01-01'),
  fechaultactualizacion date not null check (fechaultactualizacion >= date '1990-01-01'),
  fechanacimiento date not null check (fechanacimiento >= date '1920-01-01'),
  codpais bigint not null check (codpais >= 1 and codpais <= 999),
  coddepartamento bigint not null check (coddepartamento >= 1 and coddepartamento <= 99),
  codciudad bigint not null check (codciudad >= 1 and codciudad <= 99999),
  codzona bigint not null check (codzona >= 1 and codzona <= 9),
  codcomuna bigint not null check (codcomuna >= 0 and codcomuna <= 99),
  codbarrio bigint not null check (codbarrio >= 0 and codbarrio <= 9999),
  codciiu bigint not null check (codciiu >= 0 and codciiu <= 9999),
  coddivision bigint not null check (coddivision >= 0 and coddivision <= 99),
  estado text not null check (estado <> '' and estado in ('ACTIVO', 'RETIRADO')),
  estrato bigint not null check (estrato >= 1 and estrato <= 6),
  codempresa bigint not null check (codempresa >= 1),
  coddependencia bigint not null check (coddependencia >= 1),
  periododeduce text not null check (periododeduce <> '' and periododeduce in ('MENSUAL', 'QUINCENAL')),
  deduceocacional text not null check (deduceocacional <> '' and deduceocacional in ('S', 'N')),
  salario bigint not null check (salario >= 0),
  codempresalabora bigint not null check (codempresalabora >= 1),
  fechaingresolabora date not null check (fechaingresolabora >= date '1950-01-01'),
  tipocontrato text not null check (tipocontrato <> '' and tipocontrato in ('FIJO', 'INDEFINIDO')),
  fechaterminacontrato date,
  cargo text not null check (cargo <> ''),
  codbanco bigint not null check (codbanco >= 1 and codbanco <= 999),
  numerocuenta text not null check (numerocuenta <> '' and numerocuenta ~ '^[0-9]{6,20}$'),
  agencia_laboral bigint not null check (agencia_laboral >= 1 and agencia_laboral <= 99),
  codanalista bigint not null check (codanalista >= 1),
  rutaanalista text not null check (rutaanalista <> ''),
  tipocuenta text not null check (tipocuenta <> '' and tipocuenta in ('AHORROS', 'CORRIENTE')),
  estado_laboral text not null check (estado_laboral <> '' and estado_laboral in ('VIGENTE', 'RETIRADO')),
  ultimaactualizacion date not null check (ultimaactualizacion >= date '1990-01-01'),
  operadoractualizo text not null check (operadoractualizo <> ''),
  szin bigint not null check (szin >= 0),
  idtiposalario bigint not null check (idtiposalario in (1, 2)),
  tipopacto text not null check (tipopacto <> '' and tipopacto in ('CONVENCIONAL', 'INTEGRAL')),
  idasociados bigint not null check (idasociados >= 1),
  codigotipo bigint not null check (codigotipo >= 1),
  codpaisofi bigint not null check (codpaisofi >= 1 and codpaisofi <= 999),
  coddepartamentoofi bigint not null check (coddepartamentoofi >= 1 and coddepartamentoofi <= 99),
  codciudadofi bigint not null check (codciudadofi >= 1 and codciudadofi <= 99999),
  direccionempresa text not null check (direccionempresa <> ''),
  teloficina text not null check (teloficina <> '' and teloficina ~ '^[0-9]{7,10}$'),
  extension text check (extension <> '' and extension ~ '^[0-9]{1,6}$'),
  modalidadsalarial text not null check (modalidadsalarial <> '' and modalidadsalarial in ('FIJO', 'INTEGRAL')),
  num_solicitud_disminucioncuotas bigint not null check (num_solicitud_disminucioncuotas >= 0),
  num_solicitud_cancelacionahorros bigint not null check (num_solicitud_cancelacionahorros >= 0),
  num_solicitud_retiroparcialahorros bigint not null check (num_solicitud_retiroparcialahorros >= 0),
  num_cantidad_ahorrosalavista bigint not null check (num_cantidad_ahorrosalavista >= 0),
  num_cantidad_ahorroscontractuales bigint not null check (num_cantidad_ahorroscontractuales >= 0),
  cancelacioncreditos_ult3meses bigint not null check (cancelacioncreditos_ult3meses >= 0),
  num_solicitudcreditos_ult3meses bigint not null check (num_solicitudcreditos_ult3meses >= 0),
  num_cantidad_servicios bigint not null check (num_cantidad_servicios >= 0),
  num_intencionretiros_ult3meses bigint not null check (num_intencionretiros_ult3meses >= 0),
  num_retiros bigint not null check (num_retiros in (0, 1)),
  motivoretiro text check (motivoretiro <> '' and motivoretiro in ('INSATISFACCION CON TASAS', 'TERMINACION CONTRATO LABORAL', 'NECESIDAD DE LIQUIDEZ', 'TRASLADO A OTRA ENTIDAD', 'PENSION')),
  constraint retiro_coherente check ((num_retiros = 1) = (fecharetiro is not null) and (num_retiros = 1) = (estado = 'RETIRADO') and (num_retiros = 1) = (motivoretiro is not null)),
  constraint estado_laboral_coherente check ((estado = 'RETIRADO') = (estado_laboral = 'RETIRADO')),
  constraint contrato_fijo_con_fecha_fin check ((tipocontrato = 'FIJO') = (fechaterminacontrato is not null)),
  constraint fin_contrato_posterior_ingreso check (fechaterminacontrato is null or fechaterminacontrato > fechaingresolabora),
  constraint retiro_posterior_ingreso check (fecharetiro is null or fecharetiro > fechaingresolabora),
  constraint nombreintegrado_coherente check (nombreintegrado = concat_ws(' ', primerapellido, segundoapellido, nombres, segundonombre)),
  constraint mayor_de_edad_al_ingreso check (fechaingresolabora >= fechanacimiento + interval '18 years'),
  constraint pacto_integral_coherente check ((tipopacto = 'INTEGRAL') = (modalidadsalarial = 'INTEGRAL') and (tipopacto = 'INTEGRAL') = (idtiposalario = 2))
);
comment on table public.asociados_entrada is 'Entrada del formulario (contrato asociados v1.1.0)';
alter table public.asociados_entrada enable row level security;
grant insert on public.asociados_entrada to authenticated;
grant usage on sequence public.asociados_entrada_id_seq to authenticated;
drop policy if exists equipo_inserta on public.asociados_entrada;
create policy equipo_inserta on public.asociados_entrada for insert to authenticated
  with check (public.es_del_equipo() and revision = 'pendiente' and enviado_por = lower(coalesce(auth.jwt() ->> 'email', '')));
