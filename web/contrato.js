// Generado desde contratos/asociados.yaml. No editar a mano: make contrato-web
window.CONTRATO = {
  "nombre": "asociados",
  "version": "2.2.0",
  "tabla_db": "asociados",
  "grupos": {
    "SOCIODEMOGRAFICO": [
      {
        "nombre": "TIPOIDENTIFICACION",
        "tipo": "texto",
        "requerido": true,
        "db": "tipoidentificacion"
      },
      {
        "nombre": "NROIDENTIFICACION",
        "tipo": "texto",
        "requerido": true,
        "unico": true,
        "descripcion": "Identificador anonimizado; no entra al modelo",
        "db": "nroidentificacion"
      },
      {
        "nombre": "CODPAISCEDULA",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "codpaiscedula"
      },
      {
        "nombre": "PRIMERAPELLIDO",
        "tipo": "texto",
        "requerido": true,
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "primerapellido"
      },
      {
        "nombre": "SEGUNDOAPELLIDO",
        "tipo": "texto",
        "requerido": true,
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "segundoapellido"
      },
      {
        "nombre": "NOMBRES",
        "tipo": "texto",
        "requerido": true,
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "nombres"
      },
      {
        "nombre": "SEGUNDONOMBRE",
        "tipo": "texto",
        "requerido": true,
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "segundonombre"
      },
      {
        "nombre": "NOMBREINTEGRADO",
        "tipo": "texto",
        "requerido": true,
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "nombreintegrado"
      },
      {
        "nombre": "DIRECCION",
        "tipo": "texto",
        "requerido": true,
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "direccion"
      },
      {
        "nombre": "CODPAISRESIDENCIA",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "codpaisresidencia"
      },
      {
        "nombre": "CODDEPARTAMENTORESIDENCIA",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "coddepartamentoresidencia"
      },
      {
        "nombre": "CODCIUDADRESIDENCIA",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "codciudadresidencia"
      },
      {
        "nombre": "RELACION",
        "tipo": "texto",
        "requerido": true,
        "db": "relacion"
      },
      {
        "nombre": "SEXO",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "F",
          "M"
        ],
        "db": "sexo"
      },
      {
        "nombre": "EMAIL",
        "tipo": "texto",
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "email"
      },
      {
        "nombre": "FECHANACIMIENTO",
        "tipo": "fecha",
        "requerido": true,
        "min": "1920-01-01",
        "db": "fechanacimiento"
      },
      {
        "nombre": "EDAD",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "max": 120,
        "db": "edad"
      },
      {
        "nombre": "PAISNACE",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "paisnace"
      },
      {
        "nombre": "DPTONACE",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "dptonace"
      },
      {
        "nombre": "CIUDADNACE",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "ciudadnace"
      },
      {
        "nombre": "FECHAULTACTUALIZACION",
        "tipo": "fecha",
        "requerido": true,
        "min": "1990-01-01",
        "db": "fechaultactualizacion"
      },
      {
        "nombre": "ESTADO",
        "tipo": "texto",
        "requerido": true,
        "descripcion": "No entra al modelo por riesgo de fuga",
        "db": "estado"
      },
      {
        "nombre": "ESTRATO",
        "tipo": "texto",
        "valores": [
          "1",
          "2",
          "3",
          "4",
          "5",
          "6"
        ],
        "descripcion": "Opcional; el extracto real trae vacíos",
        "db": "estrato"
      },
      {
        "nombre": "CELULAR",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "descripcion": "Valor anonimizado; no entra al modelo",
        "db": "celular"
      },
      {
        "nombre": "ESTUDIOS",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "O",
          "U",
          "G",
          "S",
          "E",
          "P",
          "T"
        ],
        "db": "estudios"
      },
      {
        "nombre": "CODPROFESION",
        "tipo": "texto",
        "requerido": true,
        "db": "codprofesion"
      },
      {
        "nombre": "ESTADOCIVIL",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "C",
          "T",
          "V",
          "S",
          "D",
          "U",
          "F",
          "P"
        ],
        "db": "estadocivil"
      },
      {
        "nombre": "FECHAINGRESO",
        "tipo": "fecha",
        "requerido": true,
        "min": "1950-01-01",
        "db": "fechaingreso"
      },
      {
        "nombre": "ANTIGUEDAD_F_MESES",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "antiguedad_f_meses"
      }
    ],
    "LABORAL": [
      {
        "nombre": "CODEMPRESA",
        "tipo": "texto",
        "requerido": true,
        "db": "codempresa"
      },
      {
        "nombre": "CODDEPENDENCIA",
        "tipo": "texto",
        "requerido": true,
        "db": "coddependencia"
      },
      {
        "nombre": "SALARIO",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "descripcion": "Pesos colombianos mensuales",
        "db": "salario"
      },
      {
        "nombre": "FECHAINGRESOLABORA",
        "tipo": "fecha",
        "requerido": true,
        "db": "fechaingresolabora"
      },
      {
        "nombre": "ANTIGUEDAD_LA_MESES",
        "tipo": "entero",
        "min": 0,
        "db": "antiguedad_la_meses"
      },
      {
        "nombre": "TIPOCONTRATO",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "2",
          "1",
          "15",
          "O",
          "0"
        ],
        "db": "tipocontrato"
      },
      {
        "nombre": "CARGO",
        "tipo": "texto",
        "requerido": true,
        "db": "cargo"
      }
    ],
    "PRODUCTOS": [
      {
        "nombre": "#SOLICITUDCREDITOS_ULT3MES",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "num_solicitudcreditos_ult3mes"
      },
      {
        "nombre": "#CANTIDAD_AHORROSALAVISTA",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "num_cantidad_ahorrosalavista"
      },
      {
        "nombre": "#CANTIDAD_AHORROSCONTRACTUALES",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "num_cantidad_ahorroscontractuales"
      },
      {
        "nombre": "#SOLICITUD_RETIROPARCIALAHORROS",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "num_solicitud_retiroparcialahorros"
      },
      {
        "nombre": "#SOLICITUD_CANCELACIONAHORROS",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "num_solicitud_cancelacionahorros"
      },
      {
        "nombre": "#SOLICITUD_DISMINUCIONCUOTAS",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "SI",
          "NO"
        ],
        "db": "num_solicitud_disminucioncuotas"
      },
      {
        "nombre": "#CANTIDAD_SERVICIOS",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "db": "num_cantidad_servicios"
      }
    ],
    "RETIRO": [
      {
        "nombre": "#RETIROS",
        "tipo": "entero",
        "requerido": true,
        "min": 0,
        "descripcion": "Conteo histórico acumulado durante la permanencia; es predictor, no etiqueta futura",
        "db": "num_retiros"
      }
    ]
  }
};
