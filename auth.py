#!/usr/bin/env python3
"""auth.py — Autenticación por API key para el servidor de contrato de Whisper.

POR QUE AHORA, SI TODAVIA NO HAY NADIE AFUERA
  Regla del proyecto (filosofía §Seguridad en desarrollo vs producción): la
  seguridad es prioritaria en PRODUCCION, no en desarrollo, y un arreglo LOCAL
  puede postergarse. Pero este no es local: la credencial cambia la FORMA de
  cómo se llama al servicio, así que cada consumidor que se acople sin ella
  habría que rehacerlo. Por eso el hueco se abre ahora y los clientes empiezan a
  mandar la cabecera desde ya, aunque el servidor todavía no la exija.

APAGADA POR DEFECTO, A PROPOSITO
  Sin clientes registrados, `requerida()` es False y todo pasa igual que antes:
  no frena el desarrollo. Registrar el primer cliente la enciende. GET /health
  publica en qué estado está, para que nadie tenga que adivinar por qué recibe
  401 -- o por qué NO lo recibe.

VA EN CABECERA, NO EN EL ENVELOPE
  `Authorization: Bearer <key>`. Decisión del usuario, y el motivo es bueno: el
  envelope viaja por logs, shards y reintentos; si un tercero accede al contrato
  de alguien más, no obtiene además su credencial. Como efecto colateral, el
  contrato de job no cambia de versión por esto.

MODELO
  Mismo que PartnerClaudeProcesadorInformacion/auth.py y que
  cerebro-mcp/auth.mjs, ya construidos y validados: se guarda el SHA-256
  de la clave, nunca la clave; cada cliente tiene nombre y scope; la comparación
  es de tiempo constante. Se duplica en vez de compartirse porque el ADR-001
  prohíbe que un programa importe código de otro -- son contenedores
  independientes.

USO
  python auth.py add "bot-asignador" [--scope full|read]
  python auth.py list
  python auth.py revoke "bot-asignador"
"""
import hashlib
import hmac
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

DIR = Path(__file__).resolve().parent
AUTH_FILE = Path(os.environ.get('WHISPER_AUTH_FILE') or (DIR / '.auth' / 'clients.json'))

# Un scope 'read' solo puede consultar; las ops que producen trabajo exigen 'full'.
# El tope es por OPERACION del contrato, no por endpoint: /jobs es uno solo.
OPS_SOLO_LECTURA = frozenset()   # 'transcribe' produce trabajo: no es de solo lectura
PREFIJO_CLAVE = 'whis_'


def _hash(clave: str) -> str:
    return hashlib.sha256(clave.encode('utf-8')).hexdigest()


def _cargar() -> dict:
    try:
        return json.loads(AUTH_FILE.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _guardar(clientes: dict) -> None:
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTH_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(clientes, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, AUTH_FILE)
    try:  # el archivo de claves no debería ser legible por otros usuarios
        os.chmod(AUTH_FILE, 0o600)
    except OSError:
        pass


def requerida() -> bool:
    """True solo si hay al menos un cliente registrado. Sin clientes, el servidor
    funciona como siempre -- la seguridad no frena el desarrollo."""
    return bool(_cargar())


def agregar(nombre: str, scope: str = 'full') -> str:
    if scope not in ('full', 'read'):
        raise ValueError(f"scope inválido '{scope}' — usar 'full' o 'read'")
    clave = PREFIJO_CLAVE + secrets.token_hex(24)
    clientes = _cargar()
    clientes[_hash(clave)] = {'nombre': nombre, 'scope': scope,
                              'creado': datetime.now(timezone.utc).isoformat()}
    _guardar(clientes)
    return clave  # se devuelve UNA sola vez: después solo queda el hash


def revocar(nombre: str) -> int:
    clientes = _cargar()
    fuera = [h for h, c in clientes.items() if c.get('nombre') == nombre]
    for h in fuera:
        clientes.pop(h)
    if fuera:
        _guardar(clientes)
    return len(fuera)


def listar() -> list:
    return [{'nombre': c.get('nombre'), 'scope': c.get('scope'), 'creado': c.get('creado'),
             'hash': h[:12]} for h, c in _cargar().items()]


def verificar(cabecera: str | None, op: str | None = None):
    """(ok, cliente_o_motivo). Distingue 'falta la cabecera' de 'la clave no
    sirve': son dos errores distintos para quien llama, y confundirlos manda a
    revisar el lugar equivocado."""
    clientes = _cargar()
    if not clientes:
        return True, None                      # auth apagada
    if not cabecera:
        return False, 'falta la cabecera Authorization'
    partes = cabecera.split(None, 1)
    if len(partes) != 2 or partes[0].lower() != 'bearer':
        return False, "el formato debe ser 'Authorization: Bearer <clave>'"
    entrante = _hash(partes[1].strip())
    for h, c in clientes.items():
        # compare_digest y no ==: una comparación normal filtra información por
        # el tiempo que tarda en fallar.
        if hmac.compare_digest(h, entrante):
            if c.get('scope') == 'read' and op is not None and op not in OPS_SOLO_LECTURA:
                return False, f"el cliente '{c.get('nombre')}' tiene scope 'read' y op='{op}' escribe"
            return True, c
    return False, 'clave desconocida o revocada'


def _cli(argv) -> int:
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == 'add':
        if len(argv) < 2:
            print('uso: auth.py add "<nombre>" [--scope full|read]', file=sys.stderr)
            return 1
        scope = 'full'
        if '--scope' in argv:
            scope = argv[argv.index('--scope') + 1]
        clave = agregar(argv[1], scope)
        print(f"Cliente '{argv[1]}' creado (scope={scope}).\n\n  {clave}\n\n"
              f"Guardala ahora: solo se muestra una vez (se almacena el hash).\n"
              f"Usar como:  Authorization: Bearer {clave}\n"
              f"O en los clientes propios:  WHISPER_API_KEY={clave}")
        return 0
    if cmd == 'list':
        cs = listar()
        if not cs:
            print('(sin clientes registrados — la autenticación está APAGADA)')
            return 0
        for c in cs:
            print(f"{c['nombre']}\t{c['scope']}\t{c['creado']}\t(hash {c['hash']}...)")
        return 0
    if cmd == 'revoke':
        if len(argv) < 2:
            print('uso: auth.py revoke "<nombre>"', file=sys.stderr)
            return 1
        n = revocar(argv[1])
        print(f"{n} cliente(s) revocado(s)." + ('' if requerida() else
              ' No queda ninguno: la autenticación quedó APAGADA.'))
        return 0
    print(f"comando '{cmd}' desconocido; usar add | list | revoke", file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(_cli(sys.argv[1:]))
