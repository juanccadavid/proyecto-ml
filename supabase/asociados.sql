-- Generado desde contratos/asociados.yaml v2.0.0. No editar a mano: make contrato-sql
-- Pegar en Supabase > SQL Editor > Run. Para aplicar un contrato nuevo en desarrollo: drop table public.asociados;

create table if not exists public.asociados (
  id bigint generated always as identity primary key,
  creado_en timestamptz not null default now(),
  origen text not null default 'formulario' check (origen in ('formulario', 'script')),
  tipoidentificacion text not null check (tipoidentificacion <> ''),
  nroidentificacion text not null unique check (nroidentificacion <> ''),
  codpaiscedula bigint not null check (codpaiscedula >= 0),
  primerapellido text not null check (primerapellido <> ''),
  segundoapellido text not null check (segundoapellido <> ''),
  nombres text not null check (nombres <> ''),
  segundonombre text not null check (segundonombre <> ''),
  nombreintegrado text not null check (nombreintegrado <> ''),
  direccion text not null check (direccion <> ''),
  codpaisresidencia bigint not null check (codpaisresidencia >= 0),
  coddepartamentoresidencia bigint not null check (coddepartamentoresidencia >= 0),
  codciudadresidencia bigint not null check (codciudadresidencia >= 0),
  relacion text not null check (relacion <> ''),
  sexo text not null check (sexo <> ''),
  email text check (email <> ''),
  fechanacimiento date not null check (fechanacimiento >= date '1920-01-01'),
  edad bigint not null check (edad >= 0 and edad <= 120),
  paisnace bigint not null check (paisnace >= 0),
  dptonace bigint not null check (dptonace >= 0),
  ciudadnace bigint not null check (ciudadnace >= 0),
  fechaultactualizacion date not null check (fechaultactualizacion >= date '1990-01-01'),
  estado text not null check (estado <> ''),
  estrato text not null check (estrato <> ''),
  celular bigint not null check (celular >= 0),
  estudios text not null check (estudios <> ''),
  codprofesion text not null check (codprofesion <> ''),
  estadocivil text not null check (estadocivil <> ''),
  fechaingreso date not null check (fechaingreso >= date '1950-01-01'),
  antiguedad_f_meses bigint not null check (antiguedad_f_meses >= 0),
  codempresa text not null check (codempresa <> ''),
  coddependencia text not null check (coddependencia <> ''),
  salario bigint not null check (salario >= 0),
  fechaingresolabora date not null check (fechaingresolabora >= date '1950-01-01'),
  antiguedad_la_meses bigint check (antiguedad_la_meses >= 0),
  tipocontrato text not null check (tipocontrato <> ''),
  cargo text not null check (cargo <> ''),
  num_solicitudcreditos_ult3mes bigint not null check (num_solicitudcreditos_ult3mes >= 0),
  num_cantidad_ahorrosalavista bigint not null check (num_cantidad_ahorrosalavista >= 0),
  num_cantidad_ahorroscontractuales bigint not null check (num_cantidad_ahorroscontractuales >= 0),
  num_solicitud_retiroparcialahorros bigint not null check (num_solicitud_retiroparcialahorros >= 0),
  num_solicitud_cancelacionahorros bigint not null check (num_solicitud_cancelacionahorros >= 0),
  num_solicitud_disminucioncuotas text not null check (num_solicitud_disminucioncuotas <> ''),
  num_cantidad_servicios bigint not null check (num_cantidad_servicios >= 0),
  num_retiros bigint not null check (num_retiros >= 0),
  constraint ingreso_fondo_posterior_nacimiento check (fechaingreso > fechanacimiento),
  constraint ingreso_laboral_posterior_nacimiento check (fechaingresolabora > fechanacimiento)
);

comment on table public.asociados is 'Contrato asociados v2.0.0';
comment on column public.asociados.tipoidentificacion is 'TIPOIDENTIFICACION';
comment on column public.asociados.nroidentificacion is 'NROIDENTIFICACION — Identificador anonimizado; no entra al modelo';
comment on column public.asociados.codpaiscedula is 'CODPAISCEDULA';
comment on column public.asociados.primerapellido is 'PRIMERAPELLIDO — Valor anonimizado; no entra al modelo';
comment on column public.asociados.segundoapellido is 'SEGUNDOAPELLIDO — Valor anonimizado; no entra al modelo';
comment on column public.asociados.nombres is 'NOMBRES — Valor anonimizado; no entra al modelo';
comment on column public.asociados.segundonombre is 'SEGUNDONOMBRE — Valor anonimizado; no entra al modelo';
comment on column public.asociados.nombreintegrado is 'NOMBREINTEGRADO — Valor anonimizado; no entra al modelo';
comment on column public.asociados.direccion is 'DIRECCION — Valor anonimizado; no entra al modelo';
comment on column public.asociados.codpaisresidencia is 'CODPAISRESIDENCIA';
comment on column public.asociados.coddepartamentoresidencia is 'CODDEPARTAMENTORESIDENCIA';
comment on column public.asociados.codciudadresidencia is 'CODCIUDADRESIDENCIA';
comment on column public.asociados.relacion is 'RELACION';
comment on column public.asociados.sexo is 'SEXO';
comment on column public.asociados.email is 'EMAIL — Valor anonimizado; no entra al modelo';
comment on column public.asociados.fechanacimiento is 'FECHANACIMIENTO';
comment on column public.asociados.edad is 'EDAD';
comment on column public.asociados.paisnace is 'PAISNACE';
comment on column public.asociados.dptonace is 'DPTONACE';
comment on column public.asociados.ciudadnace is 'CIUDADNACE';
comment on column public.asociados.fechaultactualizacion is 'FECHAULTACTUALIZACION';
comment on column public.asociados.estado is 'ESTADO — No entra al modelo por riesgo de fuga';
comment on column public.asociados.estrato is 'ESTRATO';
comment on column public.asociados.celular is 'CELULAR — Valor anonimizado; no entra al modelo';
comment on column public.asociados.estudios is 'ESTUDIOS';
comment on column public.asociados.codprofesion is 'CODPROFESION';
comment on column public.asociados.estadocivil is 'ESTADOCIVIL';
comment on column public.asociados.fechaingreso is 'FECHAINGRESO';
comment on column public.asociados.antiguedad_f_meses is 'ANTIGUEDAD_F_MESES';
comment on column public.asociados.codempresa is 'CODEMPRESA';
comment on column public.asociados.coddependencia is 'CODDEPENDENCIA';
comment on column public.asociados.salario is 'SALARIO — Pesos colombianos mensuales';
comment on column public.asociados.fechaingresolabora is 'FECHAINGRESOLABORA';
comment on column public.asociados.antiguedad_la_meses is 'ANTIGUEDAD_LA_MESES';
comment on column public.asociados.tipocontrato is 'TIPOCONTRATO';
comment on column public.asociados.cargo is 'CARGO';
comment on column public.asociados.num_solicitudcreditos_ult3mes is '#SOLICITUDCREDITOS_ULT3MES';
comment on column public.asociados.num_cantidad_ahorrosalavista is '#CANTIDAD_AHORROSALAVISTA';
comment on column public.asociados.num_cantidad_ahorroscontractuales is '#CANTIDAD_AHORROSCONTRACTUALES';
comment on column public.asociados.num_solicitud_retiroparcialahorros is '#SOLICITUD_RETIROPARCIALAHORROS';
comment on column public.asociados.num_solicitud_cancelacionahorros is '#SOLICITUD_CANCELACIONAHORROS';
comment on column public.asociados.num_solicitud_disminucioncuotas is '#SOLICITUD_DISMINUCIONCUOTAS';
comment on column public.asociados.num_cantidad_servicios is '#CANTIDAD_SERVICIOS';
comment on column public.asociados.num_retiros is '#RETIROS — Conteo de retiros; el modelo deriva el evento al menos uno';

-- Seguridad: con RLS, la clave pública sola no lee, inserta, edita ni borra public.asociados. Inserta solo el equipo con
-- sesión (abajo); leer, la clave secreta (scripts subir/bajar, nunca en el HTML ni en Git) y el panel de Supabase.
alter table public.asociados enable row level security;

-- Formulario: solo el equipo puede insertar (con sesión), directo en public.asociados.
-- 1) La clave pública sola (anon, sin sesión) no inserta.
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

-- 3) Con sesión y correo del equipo: insertar (no leer, editar ni borrar) en public.asociados.
grant insert on public.asociados to authenticated;
grant usage on sequence public.asociados_id_seq to authenticated;
drop policy if exists equipo_inserta on public.asociados;
create policy equipo_inserta on public.asociados for insert to authenticated
  with check (public.es_del_equipo() and origen = 'formulario');
