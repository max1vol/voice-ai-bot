## Deployment

- Deploy the Raspberry Pi runtime with `scripts/install_pi.sh`.
- `scripts/install_pi.sh` builds a staged runtime bundle via `scripts/build_pi_deploy.py` before syncing anything to the Pi.
- Do not rsync the whole repo to the Pi. The deploy bundle is intentionally limited to runtime files needed for install and service startup.
- `watch/`, `docs/`, `tests/`, local caches, and desktop junk such as `.DS_Store` are intentionally excluded from the Pi runtime bundle.

## Verification

- Keep deploy-bundle coverage in `tests/test_deploy_bundle.py`.
- The deploy-bundle tests should prove both:
  - the staged bundle only contains runtime files, and
  - the staged bundle can be installed and byte-compiled as a Python package.
