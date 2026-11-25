# shellcheck shell=bash
# Shared setup for the packaging scripts. Source it: `. "$(dirname "$0")/../env.sh"`.
# The release packaging follows SolidCraft's (tangency-labs/solidcraft, after PhotoCraft's;
# both Apache-2.0).
#
# Exports:
#   ROOT              workspace root
#   VERSION           the app's version, from crates/spline-fitter-gui/Cargo.toml
#                     (override: SPLINE_FITTER_VERSION)
#   DIST              output directory for release artifacts (default: $ROOT/dist/release)
#   BUILD_DATE        UTC build date, YYYY-MM-DD
#   CARGO_TARGET_DIR  cargo's target dir (default: $ROOT/target)
#   APP_ID            the desktop id: Wayland app_id / X11 class, .desktop and icon name
#   BIN_NAME          the executable's name

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

# The version lives in one place: [package] version of the GUI crate.
app_version() {
  awk '
    /^\[/ { in_pkg = ($0 == "[package]"); next }
    in_pkg && $1 == "version" { gsub(/[" ]/, "", $3); print $3; exit }
  ' "$ROOT/crates/spline-fitter-gui/Cargo.toml"
}

VERSION="${SPLINE_FITTER_VERSION:-$(app_version)}"
if [ -z "$VERSION" ]; then
  echo "error: could not read the version from crates/spline-fitter-gui/Cargo.toml" >&2
  exit 1
fi
export VERSION

DIST="${DIST:-$ROOT/dist/release}"
mkdir -p "$DIST"
DIST="$(cd "$DIST" && pwd)"
export DIST

export BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%d)}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
# Must match .with_app_id(...) in crates/spline-fitter-gui/src/main.rs, or Wayland shows no icon.
export APP_ID=AirfoilSplineFitter
export BIN_NAME=spline-fitter
export APP_NAME="Airfoil Spline Fitter"
export ICON_PNG="$ROOT/crates/spline-fitter-gui/assets/app_icon.png"

warn() {
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::warning::$*"; else echo "warning: $*" >&2; fi
}

# The licence and readme into a package directory.
copy_docs() {
  local dest="$1" f
  for f in README.md LICENSE; do
    if [ -f "$ROOT/$f" ]; then cp "$ROOT/$f" "$dest/"; fi
  done
}
