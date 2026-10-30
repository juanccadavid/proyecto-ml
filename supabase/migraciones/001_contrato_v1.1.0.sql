-- Contrato asociados v1.0.0 -> v1.1.0: dos reglas nuevas entre columnas (nivel 3).
-- Para una tabla ya creada con supabase/asociados.sql v1.0.0. Pegar en Supabase > SQL Editor > Run.
-- Las reglas de distribución (nivel 2) no van en la BD: se validan por lote en Python.

alter table public.asociados
  add constraint retiro_posterior_ingreso
  check (fecharetiro is null or fecharetiro > fechaingresolabora);

alter table public.asociados
  add constraint nombreintegrado_coherente
  check (nombreintegrado = concat_ws(' ', primerapellido, segundoapellido, nombres, segundonombre));

comment on table public.asociados is 'Contrato asociados v1.1.0';
