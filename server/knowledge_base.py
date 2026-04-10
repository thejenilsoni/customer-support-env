"""
Static knowledge base, order database, account records, and billing history
used by the customer support environment. These are the ground-truth data
sources available to the agent via tool calls.
"""

# ---------------------------------------------------------------------------
# Knowledge Base Articles
# ---------------------------------------------------------------------------

KB_TOPIC_KEYWORDS = {
    "refund": ["refund", "return", "money back", "reimburse", "lost", "missing", "undelivered"],
    "account security": ["security", "breach", "hack", "suspicious", "lock", "unlock", "compromis", "unauthoriz"],
    "subscription": ["subscription", "subscript", "cancel", "renewal", "recurring", "billing cycle"],
    "billing": ["billing", "charge", "invoice", "dispute", "payment", "overcharg"],
    "credits": ["credit", "promo", "promotion", "loyalty", "discount", "voucher"],
    "escalation": ["escalat", "manager", "senior", "legal", "complaint", "unresolved"],
}

KNOWLEDGE_BASE = {
    "refund": {
        "title": "Refund Policy",
        "content": (
            "Customers may request refunds within 30 days of purchase. "
            "Orders under $100 are automatically approved without manager review. "
            "Orders $100 or above require manager sign-off. "
            "Refunds are processed within 3-5 business days back to the original payment method. "
            "For orders confirmed lost or undelivered by the carrier, refunds are issued immediately upon verification."
        ),
    },
    "account security": {
        "title": "Account Security and Suspicious Activity Response",
        "content": (
            "If suspicious activity or an unauthorized login is detected: "
            "(1) The account is automatically locked to prevent further access. "
            "(2) A security notification email is sent to the account's registered address. "
            "(3) The customer must complete identity verification before the account is unlocked. "
            "(4) The security team reviews the login history. "
            "Identity verification can be completed via the email link or by submitting a government-issued ID. "
            "Expected timeline for resolution: 24-48 hours after verification. "
            "Standard security lockouts (no confirmed breach) can be resolved directly by support agents. "
            "Confirmed data breaches must be escalated to the security team."
        ),
    },
    "subscription": {
        "title": "Subscription Cancellation and Billing",
        "content": (
            "Subscriptions cancelled before the billing date will not be charged for the next period. "
            "If a customer is charged after a confirmed cancellation date, they are entitled to a full refund "
            "of the incorrect charge. Process the refund immediately and apologize for the billing error. "
            "Cancellation confirmation emails serve as proof of cancellation date."
        ),
    },
    "billing": {
        "title": "Billing Disputes",
        "content": (
            "For billing disputes: "
            "(1) Look up all charges in the billing history. "
            "(2) Identify which charges are valid vs. erroneous based on service usage and account records. "
            "(3) Clearly explain valid charges with their source (e.g., API usage, one-time purchases). "
            "(4) Refund invalid charges within 5 business days. "
            "(5) Apply any missing credits within 24 hours of confirmation."
        ),
    },
    "credits": {
        "title": "Account Credits and Promotional Offers",
        "content": (
            "Promotional credits are applied to accounts within 3-5 business days of being issued. "
            "If a customer reports a missing credit that was promised by the support team or as part of a promotion, "
            "verify the promotion record and apply the credit manually via the account management system. "
            "Credits are non-transferable and expire 12 months after issuance."
        ),
    },
    "escalation": {
        "title": "Escalation Guidelines",
        "content": (
            "Escalate to senior support when: "
            "(1) A refund exceeds $500. "
            "(2) The customer makes legal threats. "
            "(3) A confirmed security breach is identified. "
            "(4) Multiple resolution attempts have failed. "
            "For standard security lockouts and refunds under $500, agents can resolve directly without escalation."
        ),
    },
}

# ---------------------------------------------------------------------------
# Order Database
# ---------------------------------------------------------------------------

ORDERS = {
    "ORD-1001": {
        "status": "lost_in_transit",
        "product": "Wireless Headphones Pro",
        "amount": 45.99,
        "order_date": "2026-03-11",
        "expected_delivery": "2026-03-18",
        "current_date": "2026-04-05",
        "days_since_order": 25,
        "carrier_status": "Package confirmed lost by carrier",
        "refund_eligible": True,
        "refund_auto_approved": True,
    },
    "ORD-3055": {
        "status": "cancelled",
        "product": "Pro Subscription - Monthly",
        "amount": 89.99,
        "order_date": "2026-03-01",
        "cancellation_date": "2026-03-10",
        "cancellation_confirmation": "CANC-20260310-3055",
        "charge_date": "2026-03-15",
        "notes": "Charged $89.99 on 2026-03-15 despite confirmed cancellation on 2026-03-10. INVALID CHARGE.",
        "refund_eligible": True,
        "refund_auto_approved": True,
    },
}

# ---------------------------------------------------------------------------
# Account Database
# ---------------------------------------------------------------------------

ACCOUNTS = {
    "ACC-2047": {
        "status": "locked",
        "name": "John Smith",
        "email": "jsmith@email.com",
        "lock_reason": "suspicious_login_detected",
        "lock_date": "2026-04-01",
        "suspicious_ip": "185.220.101.47 (Tor exit node, Eastern Europe)",
        "last_known_location": "San Francisco, CA, USA",
        "security_email_sent": True,
        "verification_required": True,
        "notes": "Standard lockout — no confirmed breach. Agent can resolve directly after identity verification.",
    },
    "ACC-5091": {
        "status": "active",
        "name": "Maria Wilson",
        "email": "mwilson@email.com",
        "pending_credit": 20.00,
        "credit_reason": "February service outage compensation (promised by agent on 2026-02-18)",
        "credit_applied": False,
        "credit_expiry": "2027-02-15",
        "notes": "$20 credit was promised but never applied to account. Needs manual application.",
    },
}

# ---------------------------------------------------------------------------
# Billing History
# ---------------------------------------------------------------------------

BILLING_HISTORY = {
    "ACC-5091": [
        {
            "date": "2026-03-01",
            "amount": 12.50,
            "description": "Premium API usage — February",
            "charge_id": "CHG-5091-001",
            "valid": True,
            "explanation": (
                "Customer made 250 premium API calls in February at $0.05 each. "
                "This charge is legitimate and matches usage records."
            ),
        },
        {
            "date": "2026-03-15",
            "amount": 89.99,
            "description": "Pro Subscription renewal — March",
            "charge_id": "CHG-5091-002",
            "valid": False,
            "explanation": (
                "Subscription was cancelled on 2026-03-10 (ref: CANC-20260310-3055), "
                "but the renewal was incorrectly processed on 2026-03-15. "
                "Full refund of $89.99 is owed."
            ),
        },
    ],
}
