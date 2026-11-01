import argparse, getpass, json, sys
from pathlib import Path
from b import CompactEngine

def make_engine(profile, device):
    return CompactEngine(external_nonce=profile == 'short-story', device=device, context='[短篇320字]')

def quality_cost(packet):
    t = packet['story']
    penalty = 100 * t.count('结束') + 80 * t.count('故事的寓意') + 40 * t.count('。。')
    penalty += sum((max(0, t.count(t[i:i + 5]) - 1) for i in range(len(t) - 4)))
    return len(t) + penalty

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='Character-LM story carrier: 12 or 23 public characters.')
    sub = p.add_subparsers(dest='action', required=True)
    enc = sub.add_parser('encode')
    enc.add_argument('--message', required=True)
    enc.add_argument('--received', required=True, help='Actual Beijing calendar date, YYYY-MM-DD')
    enc.add_argument('--written', required=True)
    enc.add_argument('--event', required=True)
    enc.add_argument('--profile', choices=['small-code', 'short-story'], default='small-code')
    enc.add_argument('--attempts', type=int, default=4)
    enc.add_argument('--output', type=Path, required=True)
    dec = sub.add_parser('decode')
    dec.add_argument('--packet', type=Path, required=True)
    dec.add_argument('--received', required=True)
    for q in (enc, dec):
        q.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    password = getpass.getpass('主口令（隐藏输入）：')
    if args.action == 'encode':
        if not 1 <= args.attempts <= 32:
            p.error('--attempts must be 1..32')
        e = make_engine(args.profile, args.device)
        candidates = []
        for k in range(args.attempts):
            a = e.encode(password, args.received, args.written, args.event, args.message, 4 + k % 2)
            candidates.append(a)
            e.cache.clear()
        packet = min(candidates, key=quality_cost)
        decoded = e.decode(password, args.received, packet['story'], packet['public_code'])
        if decoded['message'] != args.message:
            raise RuntimeError('Sender roundtrip failed.')
        packet['context'] = '[短篇320字]'
        packet['model_sha256'] = e.model_hash
        packet['candidate_attempts'] = args.attempts
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'packet': str(args.output.resolve()), 'verified': True, **packet}, ensure_ascii=False, indent=2))
    else:
        packet = json.loads(args.packet.read_text(encoding='utf-8'))
        profiles = {'compact-ctr64-internal-v2': 'small-code', 'compact-ctr64-external-v2': 'short-story'}
        if packet['profile'] not in profiles:
            raise ValueError('Unsupported packet profile.')
        e = make_engine(profiles[packet['profile']], args.device)
        if packet.get('model_sha256', e.model_hash) != e.model_hash:
            raise ValueError('Wrong model.')
        result = e.decode(password, args.received, packet['story'], packet['public_code'])
        print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    try:
        main()
    except ValueError as error:
        print('未通过格式、日期或认证检查：' + str(error), file=sys.stderr)
        raise SystemExit(2)
