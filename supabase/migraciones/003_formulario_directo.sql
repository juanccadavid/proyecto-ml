-- Migración 003: el equipo inserta directo en public.asociados, sin revisión (contrato v1.1.0).
-- Para una base que ya corrió la 002. Pegar en Supabase > SQL Editor > Run.
-- Generado con contrato.migracion_formulario(); una base nueva lo trae en supabase/asociados.sql.
-- Ojo: borra public.asociados_entrada (y lo que hubiera pendiente: son datos de prueba del formulario).

alter table public.asociados enable row level security;
drop table if exists public.asociados_entrada;

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
