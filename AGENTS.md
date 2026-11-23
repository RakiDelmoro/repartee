# Agent notes

## Repositories

This checkout is one of two related repositories. Do not vendor one into the other.

| Repository | Local path | Remote | Branch |
|---|---|---|---|
| `repartee` | this checkout | `https://github.com/RakiDelmoro/repartee.git` | `main` |
| `vani` | `${VANI_REPO:-../vani}` | `https://github.com/RakiDelmoro/vani.git` | `main` |

The Vani duplex research/workspace is the separate `vani` repository. `repartee`
ignores a local `vani/` directory and must never commit Vani code or artifacts.

## Updating Vani

Resolve the checkout, then inspect the real state rather than assuming it:

```bash
VANI_REPO="${VANI_REPO:-$(git rev-parse --show-toplevel)/../vani}"

git -C "$VANI_REPO" remote -v
git -C "$VANI_REPO" status -sb
git -C "$VANI_REPO" log --oneline -1
```

After editing Vani:

```bash
git -C "$VANI_REPO" add -A
git -C "$VANI_REPO" commit -m "<message>"
git -C "$VANI_REPO" push origin main
git -C "$VANI_REPO" status -sb   # must not show ahead/behind
```

Commit author: `Raki Delmoro <rakidelmoro@hotmail.com>`.

If the checkout is somewhere else, set `VANI_REPO` to its absolute path. If the
remote is missing, add it:

```bash
git -C "$VANI_REPO" remote add origin https://github.com/RakiDelmoro/vani.git
git -C "$VANI_REPO" push -u origin main
```

If HTTPS authentication fails in a new environment, use the same credential helper
as this container or switch to SSH:

```bash
git -C "$VANI_REPO" remote set-url origin git@github.com:RakiDelmoro/vani.git
```

## Updating Repartee

```bash
git status -sb
git add -A
git commit -m "<message>"
git push origin main
git status -sb   # must not show ahead/behind
```

Never commit `vani/` into this repository. The `/vani/` ignore rule is intentional.
