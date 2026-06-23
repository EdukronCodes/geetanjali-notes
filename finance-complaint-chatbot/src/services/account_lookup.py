from sqlalchemy.orm import Session

from src.database import Account, Transaction
from src.models.schemas import AccountSnapshot


def lookup_account(session: Session, account_id: str | None) -> AccountSnapshot | None:
    if not account_id:
        return None
    account = session.query(Account).filter(Account.account_id == account_id).first()
    if not account:
        return None
    txns = (
        session.query(Transaction)
        .filter(Transaction.account_id == account_id)
        .order_by(Transaction.posting_date.desc())
        .limit(5)
        .all()
    )
    return AccountSnapshot(
        account_id=account.account_id,
        customer_name=account.customer_name,
        status=account.status,
        recent_transactions=[
            {
                "transaction_id": t.transaction_id,
                "amount": t.amount,
                "currency": t.currency,
                "posting_date": str(t.posting_date),
                "description": t.description,
                "status": t.status,
            }
            for t in txns
        ],
    )


def format_account_context(snapshot: AccountSnapshot | None) -> str:
    if not snapshot:
        return ""
    lines = [
        f"Account: {snapshot.account_id} ({snapshot.customer_name}) — Status: {snapshot.status}",
        "Recent transactions:",
    ]
    for t in snapshot.recent_transactions:
        lines.append(
            f"  - {t['transaction_id']}: {t['amount']} {t['currency']} on {t['posting_date']} ({t['status']})"
        )
    return "\n".join(lines)
