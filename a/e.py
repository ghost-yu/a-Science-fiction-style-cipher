import base64
import datetime as dt
import hashlib
import hmac
import json
import struct
import sys
import time

def _date(text):
    return dt.date.fromisoformat(text)

def _prefix(key, received, encrypted_body, cover_text=''):
    context = b'TL069/date-seal/v2\x00' + received.encode('ascii') + b'\x00' + hashlib.sha256(encrypted_body).digest() + hashlib.sha256(cover_text.encode('utf-8')).digest()
    return hmac.new(key, context, hashlib.sha256)

def seal(key, received, written, event, encrypted_body, cover_text=''):
    r = _date(received)
    received = r.isoformat()
    a, b = ((_date(written) - r).days, (_date(event) - r).days)
    if not (-3650 <= a <= 3650 and -30 <= b <= 30):
        raise ValueError('Date lies outside the agreed search window; use extension format.')
    mac = _prefix(key, received, encrypted_body, cover_text)
    mac.update(struct.pack('>hh', a, b))
    return base64.urlsafe_b64encode(mac.digest()[:8]).decode('ascii').rstrip('=')

def recover(key, received, code, encrypted_body, cover_text=''):
    received = _date(received).isoformat()
    if len(code) != 11:
        raise ValueError('Date seal must have eleven characters.')
    target = base64.urlsafe_b64decode(code + '=')
    if len(target) != 8 or base64.urlsafe_b64encode(target).decode('ascii').rstrip('=') != code:
        raise ValueError('Non-canonical date seal.')
    base = _prefix(key, received, encrypted_body, cover_text)
    matches = []
    for a in range(-3650, 3651):
        for b in range(-30, 31):
            mac = base.copy()
            mac.update(struct.pack('>hh', a, b))
            if hmac.compare_digest(mac.digest()[:8], target):
                matches.append((a, b))
    if len(matches) != 1:
        raise ValueError('No unique valid date pair; wrong key/date/body or a collision.')
    a, b = matches[0]
    r = _date(received)
    return {'written': (r + dt.timedelta(days=a)).isoformat(), 'event': (r + dt.timedelta(days=b)).isoformat(), 'written_delta_days': a, 'event_delta_days': b}
