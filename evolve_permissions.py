#!/usr/bin/env python3
"""evolve_permissions.py — user-facing permission grant/revoke/check CLI.

Gap 10 Level 2 on-ramp (see docs/gap10-level2-plan.md and
docs/gap10-bridge-design.md). The design's Level 2 gate is an EXPLICIT
USER GRANT before any write api_call may pass pre-flight; this tool is
the channel through which that grant is issued. It is also useful for
auditing the live registry at any time.

All mutations go through data_layer.SelfModel (load -> mutate -> save)
so the daemon and the host bridge see exactly the same registry, written
atomically. Deny-by-default is preserved: an entry only ever appears via
grant_permission (full False matrix + cap first), and revoke flips a
single flag back to False.

Usage:
  python3 evolve_permissions.py show [resource]
  python3 evolve_permissions.py grant <resource> <read|write|act> [cap] [--yes]
  python3 evolve_permissions.py revoke <resource> <read|write|act>
  python3 evolve_permissions.py check <resource> <read|write|act>   # exit 0 iff granted

Exit codes: 0 = success/granted; 1 = denied / nothing changed;
2 = usage or validation error.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from data_layer import SelfModel

_ACTIONS = ("read", "write", "act")
_USAGE = (
    "usage: evolve_permissions.py {show|grant|revoke|check} ... "
    "(run with --help for details)"
)


def _fmt(entry: Optional[dict]) -> str:
    if not isinstance(entry, dict):
        return str(entry)
    flags = ", ".join(a for a in _ACTIONS if entry.get(a))
    cap = entry.get("cap")
    return "{" + f"granted: {flags or '(none)'}, cap: {cap!r}" + "}"


def _load() -> SelfModel:
    return SelfModel.load()


def _confirm(prompt: str) -> Optional[bool]:
    """Interactive confirmation for a grant; --yes skips this in scripts.

    Returns True/False for y/n, or None when confirmation is impossible
    (no tty) so the caller can fail with a distinct exit code.
    """
    try:
        ans = input(f"{prompt} [y/N] ")
    except EOFError:
        print("aborted: no tty — pass --yes to confirm non-interactively",
              file=sys.stderr)
        return None
    return ans.strip().lower() in ("y", "yes")


def cmd_show(args: argparse.Namespace) -> int:
    sm = _load()
    perms = sm.permissions
    if args.resource:
        entry = perms.get(args.resource)
        if entry is None:
            print(f"{args.resource}: (no entry — deny by default)")
            return 1
        print(f"{args.resource}: {_fmt(entry)}")
        return 0
    if not perms:
        print("(no permission entries — deny by default)")
    for res in sorted(perms):
        print(f"{res}: {_fmt(perms[res])}")
    return 0


def cmd_grant(args: argparse.Namespace) -> int:
    if args.cap is not None and args.cap < 0:
        print(f"ERROR: cap must be non-negative, got {args.cap!r}",
              file=sys.stderr)
        return 2
    if not args.yes:
        confirmed = _confirm(
            f"Grant {args.resource}.{args.action}?"
            + (f" (cap={args.cap})" if args.cap is not None else "")
        )
        if confirmed is None:
            return 2  # cannot confirm non-interactively
        if not confirmed:
            print("aborted")
            return 1
    sm = _load()
    before = sm.permission_entry(args.resource)
    entry = sm.grant_permission(args.resource, args.action, cap=args.cap)
    problems = sm.validate_permissions()
    if problems:
        print("ERROR: registry invalid after grant — NOT saved:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2
    sm.save()
    print(f"granted {args.resource}.{args.action}")
    print(f"  before: {_fmt(before)}")
    print(f"  after:  {_fmt(entry)}")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    sm = _load()
    flipped = sm.revoke_permission(args.resource, args.action)
    sm.save()
    if flipped:
        print(f"revoked {args.resource}.{args.action}")
        return 0
    print(f"{args.resource}.{args.action} was already unset (no change)")
    return 1


def cmd_check(args: argparse.Namespace) -> int:
    sm = _load()
    ok = sm.check_permission(args.resource, args.action)
    print(f"{args.resource}.{args.action}: {'GRANTED' if ok else 'denied'}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evolve_permissions.py",
        description="Grant / revoke / check the daemon's external-action "
                    "permission registry (Gap 10 Level 2 on-ramp).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("show", help="show the permission registry")
    sp.add_argument("resource", nargs="?", default=None,
                    help="show only this resource (default: all)")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("grant", help="explicitly grant an action on a resource")
    sp.add_argument("resource")
    sp.add_argument("action", choices=_ACTIONS)
    sp.add_argument("cap", nargs="?", type=float, default=None,
                    help="optional value cap (Level 3+; default: uncapped)")
    sp.add_argument("--yes", action="store_true",
                    help="skip the interactive confirmation")
    sp.set_defaults(func=cmd_grant)

    sp = sub.add_parser("revoke", help="revoke an action on a resource")
    sp.add_argument("resource")
    sp.add_argument("action", choices=_ACTIONS)
    sp.set_defaults(func=cmd_revoke)

    sp = sub.add_parser("check", help="exit 0 iff the action is granted")
    sp.add_argument("resource")
    sp.add_argument("action", choices=_ACTIONS)
    sp.set_defaults(func=cmd_check)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
