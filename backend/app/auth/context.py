"""The authorization context.

One immutable object carries who is asking and which patient they may see.
It is created once per request from the verified JWT and then threaded
explicitly through every service, tool and retrieval call.

Explicit threading is the point. A context stored in a ``ContextVar`` or on
the request object can be forgotten; a required parameter cannot. Every
data-access function in :mod:`app.services` takes an ``AuthContext`` as its
first argument and derives the patient scope from it, so there is no
overload where a caller — or a tool schema exposed to the model — can pass
a patient id of its own choosing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["patient", "clinician", "admin"]


class AuthorizationError(Exception):
    """Raised when a context cannot satisfy the scope an operation needs.

    Deliberately not an ``HTTPException``: services are also called from the
    MCP server and the evaluation runner, neither of which speaks HTTP. The
    API layer translates this into a 403.
    """


@dataclass(frozen=True, slots=True)
class AuthContext:
    user_id: int
    role: Role
    #: The single patient this session may read. ``None`` for accounts not
    #: bound to a patient record; such a session can reach no clinical data
    #: in this demo.
    patient_id: int | None = None
    #: Correlates logs, traces and audit rows for one inbound request.
    request_id: str = field(default="")

    @property
    def patient_scope(self) -> int:
        """The patient id to filter by, or raise.

        Every query that touches clinical data goes through this property
        rather than reading ``patient_id`` directly, so a missing scope
        fails loudly instead of quietly producing an unfiltered query.
        """
        if self.patient_id is None:
            raise AuthorizationError(
                "This account is not linked to a patient record."
            )
        return self.patient_id

    def assert_patient(self, requested_patient_id: int) -> int:
        """Check a caller-supplied patient id against the session's scope.

        Used by the ``/api/patients/{id}/...`` routes, where the id is in
        the URL. A mismatch is a refusal, never a silent redirect to the
        caller's own record: quietly rewriting the id would hide probing.
        """
        scope = self.patient_scope
        if requested_patient_id != scope:
            raise AuthorizationError(
                "Not authorized to access records for this patient."
            )
        return scope
