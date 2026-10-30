-- Generado desde contratos/asociados.yaml v1.0.0. No editar a mano: make contrato-sql
-- Pegar en Supabase > SQL Editor > Run. Para aplicar un contrato nuevo en desarrollo: drop table public.asociados;

create table if not exists public.asociados (
  id bigint generated always as identity primary key,
  creado_en timestamptz not null default now(),
  origen text not null default 'formulario' check (origen in ('formulario', 'script')),
  cedula bigint not null unique check (cedula >= 1 and cedula <= 9999999999),
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
  idasociados bigint not null unique check (idasociados >= 1),
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
  constraint mayor_de_edad_al_ingreso check (fechaingresolabora >= fechanacimiento + interval '18 years'),
  constraint pacto_integral_coherente check ((tipopacto = 'INTEGRAL') = (modalidadsalarial = 'INTEGRAL') and (tipopacto = 'INTEGRAL') = (idtiposalario = 2))
);

comment on table public.asociados is 'Contrato asociados v1.0.0';
comment on column public.asociados.cedula is 'CEDULA — Número de documento';
comment on column public.asociados.primerapellido is 'PRIMERAPELLIDO';
comment on column public.asociados.segundoapellido is 'SEGUNDOAPELLIDO';
comment on column public.asociados.nombres is 'NOMBRES';
comment on column public.asociados.segundonombre is 'SEGUNDONOMBRE';
comment on column public.asociados.nombreintegrado is 'NOMBREINTEGRADO';
comment on column public.asociados.direccion is 'DIRECCION';
comment on column public.asociados.tipoidentificacion is 'TIPOIDENTIFICACION';
comment on column public.asociados.relacion is 'RELACION';
comment on column public.asociados.agencia is 'AGENCIA';
comment on column public.asociados.telefono1 is 'TELEFONO1';
comment on column public.asociados.extension1 is 'EXTENSION1';
comment on column public.asociados.telefono2 is 'TELEFONO2';
comment on column public.asociados.extension2 is 'EXTENSION2';
comment on column public.asociados.apartadoaereo is 'APARTADOAEREO';
comment on column public.asociados.sexo is 'SEXO';
comment on column public.asociados.codpaiscedula is 'CODPAISCEDULA';
comment on column public.asociados.codciudadcedula is 'CODCIUDADCEDULA — Código DANE';
comment on column public.asociados.email is 'EMAIL';
comment on column public.asociados.fecharetiro is 'FECHARETIRO — Solo si el asociado se retiró';
comment on column public.asociados.fechaultactualizacion is 'FECHAULTACTUALIZACION';
comment on column public.asociados.fechanacimiento is 'FECHANACIMIENTO';
comment on column public.asociados.codpais is 'CODPAIS';
comment on column public.asociados.coddepartamento is 'CODDEPARTAMENTO';
comment on column public.asociados.codciudad is 'CODCIUDAD — Código DANE';
comment on column public.asociados.codzona is 'CODZONA';
comment on column public.asociados.codcomuna is 'CODCOMUNA — 0 fuera de Medellín';
comment on column public.asociados.codbarrio is 'CODBARRIO — 0 fuera de Medellín';
comment on column public.asociados.codciiu is 'CODCIIU';
comment on column public.asociados.coddivision is 'CODDIVISION';
comment on column public.asociados.estado is 'ESTADO';
comment on column public.asociados.estrato is 'ESTRATO';
comment on column public.asociados.codempresa is 'CODEMPRESA';
comment on column public.asociados.coddependencia is 'CODDEPENDENCIA';
comment on column public.asociados.periododeduce is 'PERIODODEDUCE';
comment on column public.asociados.deduceocacional is 'DEDUCEOCACIONAL';
comment on column public.asociados.salario is 'SALARIO — Pesos colombianos mensuales';
comment on column public.asociados.codempresalabora is 'CODEMPRESALABORA';
comment on column public.asociados.fechaingresolabora is 'FECHAINGRESOLABORA';
comment on column public.asociados.tipocontrato is 'TIPOCONTRATO';
comment on column public.asociados.fechaterminacontrato is 'FECHATERMINACONTRATO — Solo para contrato FIJO';
comment on column public.asociados.cargo is 'CARGO';
comment on column public.asociados.codbanco is 'CODBANCO';
comment on column public.asociados.numerocuenta is 'NUMEROCUENTA';
comment on column public.asociados.agencia_laboral is 'AGENCIA_LABORAL — AGENCIA del bloque LABORAL';
comment on column public.asociados.codanalista is 'CODANALISTA';
comment on column public.asociados.rutaanalista is 'RUTAANALISTA';
comment on column public.asociados.tipocuenta is 'TIPOCUENTA';
comment on column public.asociados.estado_laboral is 'ESTADO_LABORAL — ESTADO del bloque LABORAL';
comment on column public.asociados.ultimaactualizacion is 'ULTIMAACTUALIZACION';
comment on column public.asociados.operadoractualizo is 'OPERADORACTUALIZO';
comment on column public.asociados.szin is 'SZIN';
comment on column public.asociados.idtiposalario is 'IDTIPOSALARIO — 1 convencional, 2 integral';
comment on column public.asociados.tipopacto is 'TIPOPACTO';
comment on column public.asociados.idasociados is 'IDASOCIADOS';
comment on column public.asociados.codigotipo is 'CODIGOTIPO';
comment on column public.asociados.codpaisofi is 'CODPAISOFI';
comment on column public.asociados.coddepartamentoofi is 'CODDEPARTAMENTOOFI';
comment on column public.asociados.codciudadofi is 'CODCIUDADOFI';
comment on column public.asociados.direccionempresa is 'DIRECCIONEMPRESA';
comment on column public.asociados.teloficina is 'TELOFICINA';
comment on column public.asociados.extension is 'EXTENSION';
comment on column public.asociados.modalidadsalarial is 'MODALIDADSALARIAL';
comment on column public.asociados.num_solicitud_disminucioncuotas is '#SOLICITUD_DISMINUCIONCUOTAS';
comment on column public.asociados.num_solicitud_cancelacionahorros is '#SOLICITUD_CANCELACIONAHORROS';
comment on column public.asociados.num_solicitud_retiroparcialahorros is '#SOLICITUD_RETIROPARCIALAHORROS';
comment on column public.asociados.num_cantidad_ahorrosalavista is '#CANTIDAD_AHORROSALAVISTA';
comment on column public.asociados.num_cantidad_ahorroscontractuales is '#CANTIDAD_AHORROSCONTRACTUALES';
comment on column public.asociados.cancelacioncreditos_ult3meses is 'CANCELACIONCREDITOS_ULT3MESES';
comment on column public.asociados.num_solicitudcreditos_ult3meses is '#SOLICITUDCREDITOS_ULT3MESES';
comment on column public.asociados.num_cantidad_servicios is '#CANTIDAD_SERVICIOS';
comment on column public.asociados.num_intencionretiros_ult3meses is '#INTENCIONRETIROS_ULT3MESES';
comment on column public.asociados.num_retiros is '#RETIROS — Objetivo: 1 si se retiró';
comment on column public.asociados.motivoretiro is 'MOTIVORETIRO — Solo si el asociado se retiró';

-- Seguridad: la clave pública (anon/publishable) solo puede INSERTAR. Leer, editar y borrar
-- queda para el panel de Supabase o la clave service_role (que nunca va en el HTML ni en Git).
alter table public.asociados enable row level security;
grant insert on public.asociados to anon;
grant usage on sequence public.asociados_id_seq to anon;
drop policy if exists anon_solo_inserta on public.asociados;
create policy anon_solo_inserta on public.asociados for insert to anon with check (true);
