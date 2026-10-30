// Generado desde contratos/asociados.yaml. No editar a mano: make contrato-web
window.CONTRATO = {
  "nombre": "asociados",
  "version": "1.1.0",
  "tabla_db": "asociados",
  "grupos": {
    "SOCIODEMOGRAFICO": [
      {
        "nombre": "CEDULA",
        "tipo": "entero",
        "requerido": true,
        "unico": true,
        "min": 1,
        "max": 9999999999,
        "descripcion": "Número de documento",
        "db": "cedula"
      },
      {
        "nombre": "PRIMERAPELLIDO",
        "tipo": "texto",
        "requerido": true,
        "db": "primerapellido"
      },
      {
        "nombre": "SEGUNDOAPELLIDO",
        "tipo": "texto",
        "db": "segundoapellido"
      },
      {
        "nombre": "NOMBRES",
        "tipo": "texto",
        "requerido": true,
        "db": "nombres"
      },
      {
        "nombre": "SEGUNDONOMBRE",
        "tipo": "texto",
        "db": "segundonombre"
      },
      {
        "nombre": "NOMBREINTEGRADO",
        "tipo": "texto",
        "requerido": true,
        "db": "nombreintegrado"
      },
      {
        "nombre": "DIRECCION",
        "tipo": "texto",
        "requerido": true,
        "db": "direccion"
      },
      {
        "nombre": "TIPOIDENTIFICACION",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "CC",
          "CE",
          "TI",
          "PA"
        ],
        "db": "tipoidentificacion"
      },
      {
        "nombre": "RELACION",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "ASOCIADO"
        ],
        "db": "relacion"
      },
      {
        "nombre": "AGENCIA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 99,
        "db": "agencia"
      },
      {
        "nombre": "TELEFONO1",
        "tipo": "texto",
        "requerido": true,
        "patron": "^[0-9]{7,10}$",
        "db": "telefono1"
      },
      {
        "nombre": "EXTENSION1",
        "tipo": "texto",
        "patron": "^[0-9]{1,6}$",
        "db": "extension1"
      },
      {
        "nombre": "TELEFONO2",
        "tipo": "texto",
        "patron": "^[0-9]{7,10}$",
        "db": "telefono2"
      },
      {
        "nombre": "EXTENSION2",
        "tipo": "texto",
        "patron": "^[0-9]{1,6}$",
        "db": "extension2"
      },
      {
        "nombre": "APARTADOAEREO",
        "tipo": "texto",
        "db": "apartadoaereo"
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
        "nombre": "CODPAISCEDULA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 999,
        "db": "codpaiscedula"
      },
      {
        "nombre": "CODCIUDADCEDULA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 99999,
        "descripcion": "Código DANE",
        "db": "codciudadcedula"
      },
      {
        "nombre": "EMAIL",
        "tipo": "texto",
        "requerido": true,
        "patron": "^[^@ ]+@[^@ ]+[.][^@ ]+$",
        "db": "email"
      },
      {
        "nombre": "FECHARETIRO",
        "tipo": "fecha",
        "min": "1990-01-01",
        "descripcion": "Solo si el asociado se retiró",
        "db": "fecharetiro"
      },
      {
        "nombre": "FECHAULTACTUALIZACION",
        "tipo": "fecha",
        "requerido": true,
        "min": "1990-01-01",
        "db": "fechaultactualizacion"
      },
      {
        "nombre": "FECHANACIMIENTO",
        "tipo": "fecha",
        "requerido": true,
        "min": "1920-01-01",
        "db": "fechanacimiento"
      },
      {
        "nombre": "CODPAIS",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 999,
        "db": "codpais"
      },
      {
        "nombre": "CODDEPARTAMENTO",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 99,
        "db": "coddepartamento"
      },
      {
        "nombre": "CODCIUDAD",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 99999,
        "descripcion": "Código DANE",
        "db": "codciudad"
      },
      {
        "nombre": "CODZONA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 9,
        "db": "codzona"
      },
      {
        "nombre": "CODCOMUNA",
        "tipo": "entero",
        "requerido": true,
        "max": 99,
        "descripcion": "0 fuera de Medellín",
        "db": "codcomuna"
      },
      {
        "nombre": "CODBARRIO",
        "tipo": "entero",
        "requerido": true,
        "max": 9999,
        "descripcion": "0 fuera de Medellín",
        "db": "codbarrio"
      },
      {
        "nombre": "CODCIIU",
        "tipo": "entero",
        "requerido": true,
        "max": 9999,
        "db": "codciiu"
      },
      {
        "nombre": "CODDIVISION",
        "tipo": "entero",
        "requerido": true,
        "max": 99,
        "db": "coddivision"
      },
      {
        "nombre": "ESTADO",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "ACTIVO",
          "RETIRADO"
        ],
        "db": "estado"
      },
      {
        "nombre": "ESTRATO",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 6,
        "db": "estrato"
      }
    ],
    "LABORAL": [
      {
        "nombre": "CODEMPRESA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "db": "codempresa"
      },
      {
        "nombre": "CODDEPENDENCIA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "db": "coddependencia"
      },
      {
        "nombre": "PERIODODEDUCE",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "MENSUAL",
          "QUINCENAL"
        ],
        "db": "periododeduce"
      },
      {
        "nombre": "DEDUCEOCACIONAL",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "S",
          "N"
        ],
        "db": "deduceocacional"
      },
      {
        "nombre": "SALARIO",
        "tipo": "entero",
        "requerido": true,
        "descripcion": "Pesos colombianos mensuales",
        "db": "salario"
      },
      {
        "nombre": "CODEMPRESALABORA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "db": "codempresalabora"
      },
      {
        "nombre": "FECHAINGRESOLABORA",
        "tipo": "fecha",
        "requerido": true,
        "min": "1950-01-01",
        "db": "fechaingresolabora"
      },
      {
        "nombre": "TIPOCONTRATO",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "FIJO",
          "INDEFINIDO"
        ],
        "db": "tipocontrato"
      },
      {
        "nombre": "FECHATERMINACONTRATO",
        "tipo": "fecha",
        "descripcion": "Solo para contrato FIJO",
        "db": "fechaterminacontrato"
      },
      {
        "nombre": "CARGO",
        "tipo": "texto",
        "requerido": true,
        "db": "cargo"
      },
      {
        "nombre": "CODBANCO",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 999,
        "db": "codbanco"
      },
      {
        "nombre": "NUMEROCUENTA",
        "tipo": "texto",
        "requerido": true,
        "patron": "^[0-9]{6,20}$",
        "db": "numerocuenta"
      },
      {
        "nombre": "AGENCIA_LABORAL",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 99,
        "descripcion": "AGENCIA del bloque LABORAL",
        "db": "agencia_laboral"
      },
      {
        "nombre": "CODANALISTA",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "db": "codanalista"
      },
      {
        "nombre": "RUTAANALISTA",
        "tipo": "texto",
        "requerido": true,
        "db": "rutaanalista"
      },
      {
        "nombre": "TIPOCUENTA",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "AHORROS",
          "CORRIENTE"
        ],
        "db": "tipocuenta"
      },
      {
        "nombre": "ESTADO_LABORAL",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "VIGENTE",
          "RETIRADO"
        ],
        "descripcion": "ESTADO del bloque LABORAL",
        "db": "estado_laboral"
      },
      {
        "nombre": "ULTIMAACTUALIZACION",
        "tipo": "fecha",
        "requerido": true,
        "min": "1990-01-01",
        "db": "ultimaactualizacion"
      },
      {
        "nombre": "OPERADORACTUALIZO",
        "tipo": "texto",
        "requerido": true,
        "db": "operadoractualizo"
      },
      {
        "nombre": "SZIN",
        "tipo": "entero",
        "requerido": true,
        "db": "szin"
      },
      {
        "nombre": "IDTIPOSALARIO",
        "tipo": "entero",
        "requerido": true,
        "valores": [
          1,
          2
        ],
        "descripcion": "1 convencional, 2 integral",
        "db": "idtiposalario"
      },
      {
        "nombre": "TIPOPACTO",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "CONVENCIONAL",
          "INTEGRAL"
        ],
        "db": "tipopacto"
      },
      {
        "nombre": "IDASOCIADOS",
        "tipo": "entero",
        "requerido": true,
        "unico": true,
        "min": 1,
        "db": "idasociados"
      },
      {
        "nombre": "CODIGOTIPO",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "db": "codigotipo"
      },
      {
        "nombre": "CODPAISOFI",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 999,
        "db": "codpaisofi"
      },
      {
        "nombre": "CODDEPARTAMENTOOFI",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 99,
        "db": "coddepartamentoofi"
      },
      {
        "nombre": "CODCIUDADOFI",
        "tipo": "entero",
        "requerido": true,
        "min": 1,
        "max": 99999,
        "db": "codciudadofi"
      },
      {
        "nombre": "DIRECCIONEMPRESA",
        "tipo": "texto",
        "requerido": true,
        "db": "direccionempresa"
      },
      {
        "nombre": "TELOFICINA",
        "tipo": "texto",
        "requerido": true,
        "patron": "^[0-9]{7,10}$",
        "db": "teloficina"
      },
      {
        "nombre": "EXTENSION",
        "tipo": "texto",
        "patron": "^[0-9]{1,6}$",
        "db": "extension"
      },
      {
        "nombre": "MODALIDADSALARIAL",
        "tipo": "texto",
        "requerido": true,
        "valores": [
          "FIJO",
          "INTEGRAL"
        ],
        "db": "modalidadsalarial"
      }
    ],
    "PRODUCTOS": [
      {
        "nombre": "#SOLICITUD_DISMINUCIONCUOTAS",
        "tipo": "entero",
        "requerido": true,
        "db": "num_solicitud_disminucioncuotas"
      },
      {
        "nombre": "#SOLICITUD_CANCELACIONAHORROS",
        "tipo": "entero",
        "requerido": true,
        "db": "num_solicitud_cancelacionahorros"
      },
      {
        "nombre": "#SOLICITUD_RETIROPARCIALAHORROS",
        "tipo": "entero",
        "requerido": true,
        "db": "num_solicitud_retiroparcialahorros"
      },
      {
        "nombre": "#CANTIDAD_AHORROSALAVISTA",
        "tipo": "entero",
        "requerido": true,
        "db": "num_cantidad_ahorrosalavista"
      },
      {
        "nombre": "#CANTIDAD_AHORROSCONTRACTUALES",
        "tipo": "entero",
        "requerido": true,
        "db": "num_cantidad_ahorroscontractuales"
      },
      {
        "nombre": "CANCELACIONCREDITOS_ULT3MESES",
        "tipo": "entero",
        "requerido": true,
        "db": "cancelacioncreditos_ult3meses"
      },
      {
        "nombre": "#SOLICITUDCREDITOS_ULT3MESES",
        "tipo": "entero",
        "requerido": true,
        "db": "num_solicitudcreditos_ult3meses"
      },
      {
        "nombre": "#CANTIDAD_SERVICIOS",
        "tipo": "entero",
        "requerido": true,
        "db": "num_cantidad_servicios"
      }
    ],
    "RETIRO": [
      {
        "nombre": "#INTENCIONRETIROS_ULT3MESES",
        "tipo": "entero",
        "requerido": true,
        "db": "num_intencionretiros_ult3meses"
      },
      {
        "nombre": "#RETIROS",
        "tipo": "entero",
        "requerido": true,
        "valores": [
          0,
          1
        ],
        "descripcion": "Objetivo: 1 si se retiró",
        "db": "num_retiros"
      },
      {
        "nombre": "MOTIVORETIRO",
        "tipo": "texto",
        "valores": [
          "INSATISFACCION CON TASAS",
          "TERMINACION CONTRATO LABORAL",
          "NECESIDAD DE LIQUIDEZ",
          "TRASLADO A OTRA ENTIDAD",
          "PENSION"
        ],
        "descripcion": "Solo si el asociado se retiró",
        "db": "motivoretiro"
      }
    ]
  }
};
