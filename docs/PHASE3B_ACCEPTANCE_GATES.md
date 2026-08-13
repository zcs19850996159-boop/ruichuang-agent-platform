# Phase 3B acceptance gates

## Member and credential lifecycle

- Tenant owners and administrators can list members.
- Only an owner can grant or remove the `owner` role.
- The final active owner cannot be demoted or disabled.
- Disabled memberships cannot authenticate even when their token is active.
- Token list responses never contain token hashes or plaintext tokens.
- Token revocation takes effect on the next request.

## Knowledge administration visibility

- Knowledge-space lists are tenant-scoped.
- Published version and active-pointer queries require `knowledge:read`.
- Version metadata never exposes server filesystem paths.
- A caller cannot infer whether another tenant's version or space exists.

## Administration console

- The console contains no embedded credentials.
- API tokens are kept only in page memory and are cleared on disconnect.
- Untrusted names and audit details are rendered with `textContent`, not HTML.
- All data requests remain protected by the control API.

## Compatibility

- Phase 1, Phase 2, and Phase 3A tests remain green.
- Competition `/chat` behavior remains unchanged.
- The existing Phase 2 production service is not replaced during Phase 3B
  acceptance.
