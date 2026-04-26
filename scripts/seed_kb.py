"""
Seed the knowledge base vector store with initial documents.

Run: python scripts/seed_kb.py
     python scripts/seed_kb.py --clear   # wipe existing index first
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.vector_store import VectorDocument, VectorStore

DOCUMENTS = [
    # ── Returns & Refunds ────────────────────────────────────────────────
    VectorDocument(
        id="kb-returns-001",
        text=(
            "Return Policy: Customers may return items within 30 days of delivery for a full refund. "
            "Items must be in original, unused condition with original packaging. "
            "Digital products and downloadable software are non-refundable. "
            "Sale items can only be refunded for store credit. "
            "To initiate a return, contact support with your order number."
        ),
        metadata={"category": "returns", "last_updated": "2025-01-15", "author": "policy_team"},
    ),
    VectorDocument(
        id="kb-returns-002",
        text=(
            "Refund Processing Times: Once a return is received and inspected, refunds are "
            "processed within 3-5 business days. Credit card refunds appear within 5-10 business "
            "days depending on your bank. PayPal refunds are typically instant. Store credit is "
            "applied immediately. You will receive an email confirmation when the refund is initiated."
        ),
        metadata={"category": "returns", "last_updated": "2025-01-15"},
    ),
    VectorDocument(
        id="kb-returns-003",
        text=(
            "Damaged or Defective Items: If you receive a damaged or defective item, contact us "
            "within 7 days of delivery. We will arrange a free return label and send a replacement "
            "at no charge. Photos of the damage may be required. For high-value items (over $200), "
            "a pickup may be scheduled instead of a mail return."
        ),
        metadata={"category": "returns", "last_updated": "2025-02-01"},
    ),
    # ── Shipping ─────────────────────────────────────────────────────────
    VectorDocument(
        id="kb-shipping-001",
        text=(
            "Shipping Options: Standard shipping (5-7 business days) is free for orders over $50, "
            "otherwise $5.99. Express shipping (2-3 business days) costs $12.99. "
            "Overnight shipping (next business day) is $24.99 for orders placed before 2pm ET Monday-Friday. "
            "International shipping is available to 50+ countries; rates calculated at checkout. "
            "Orders are typically processed within 1 business day."
        ),
        metadata={"category": "shipping", "last_updated": "2025-03-01"},
    ),
    VectorDocument(
        id="kb-shipping-002",
        text=(
            "Order Tracking: Once your order ships, you will receive a tracking number by email. "
            "You can track your order on our website under 'My Orders' or directly on the carrier's "
            "website. Tracking updates may take 24 hours to appear after shipment. If your tracking "
            "shows 'delivered' but you haven't received the package, wait 24 hours — carriers "
            "sometimes mark packages early — then contact us."
        ),
        metadata={"category": "shipping", "last_updated": "2025-03-01"},
    ),
    VectorDocument(
        id="kb-shipping-003",
        text=(
            "Missing or Lost Packages: If your package hasn't arrived 3 business days after the "
            "estimated delivery date, contact our support team. We will file a trace with the carrier. "
            "If the package is confirmed lost, we will send a free replacement or issue a full refund. "
            "Claims for lost packages must be filed within 30 days of the original ship date."
        ),
        metadata={"category": "shipping", "last_updated": "2025-03-15"},
    ),
    # ── Billing & Payments ────────────────────────────────────────────────
    VectorDocument(
        id="kb-billing-001",
        text=(
            "Payment Methods: We accept Visa, Mastercard, American Express, Discover, PayPal, "
            "Apple Pay, Google Pay, and bank transfers for orders over $1000. "
            "All transactions are encrypted with TLS 1.3. "
            "We do not store credit card numbers — payments are processed by Stripe. "
            "Invoices for enterprise customers are available with net-30 terms upon approval."
        ),
        metadata={"category": "billing", "last_updated": "2025-02-10"},
    ),
    VectorDocument(
        id="kb-billing-002",
        text=(
            "Subscription Billing: Pro and Enterprise plans are billed monthly or annually. "
            "Annual plans receive a 20% discount. Billing occurs on the same date each period. "
            "If your card is declined, we retry for 3 days before suspending the account. "
            "You will receive email notifications 7 days and 1 day before renewal. "
            "To cancel, go to Account > Subscription > Cancel — cancellation takes effect at "
            "the end of the current billing period."
        ),
        metadata={"category": "billing", "last_updated": "2025-03-20"},
    ),
    VectorDocument(
        id="kb-billing-003",
        text=(
            "Invoice Disputes: If you believe you were charged incorrectly, contact billing support "
            "within 60 days of the charge. Provide your order number or invoice ID and a description "
            "of the discrepancy. We resolve billing disputes within 5 business days. Duplicate charges "
            "are refunded immediately upon verification."
        ),
        metadata={"category": "billing", "last_updated": "2025-01-20"},
    ),
    # ── Warranty ─────────────────────────────────────────────────────────
    VectorDocument(
        id="kb-warranty-001",
        text=(
            "Warranty Information: All hardware products carry a 2-year manufacturer warranty against defects. "
            "Accidental damage is not covered under warranty. "
            "To file a warranty claim, provide proof of purchase and a description of the defect. "
            "Warranty replacements are shipped within 5 business days of approval. "
            "Extended warranty plans are available for purchase within 90 days of original purchase."
        ),
        metadata={"category": "warranty", "last_updated": "2024-11-20"},
    ),
    # ── Account & Security ───────────────────────────────────────────────
    VectorDocument(
        id="kb-account-001",
        text=(
            "Password Reset: To reset your password, click 'Forgot Password' on the login page "
            "and enter your email address. You will receive a reset link valid for 30 minutes. "
            "If you don't receive the email within 5 minutes, check your spam folder or contact "
            "support. For security, we never send passwords via email — only reset links."
        ),
        metadata={"category": "account", "last_updated": "2025-01-01"},
    ),
    VectorDocument(
        id="kb-account-002",
        text=(
            "Account Deletion & GDPR: You may request deletion of your account and personal data "
            "at any time by emailing privacy@company.com or submitting a request in Account Settings. "
            "Data deletion is completed within 30 days. Financial records may be retained for 7 years "
            "as required by law. You will receive confirmation once deletion is complete."
        ),
        metadata={"category": "account", "last_updated": "2025-01-01"},
    ),
    # ── Technical / Product ────────────────────────────────────────────────
    VectorDocument(
        id="kb-technical-001",
        text=(
            "Compatibility Requirements: Widget Pro requires Windows 10+ or macOS 12+. "
            "Minimum 8GB RAM and 4GB free disk space. Internet connection required for activation. "
            "Not compatible with 32-bit operating systems. For Linux, use our web application "
            "at app.company.com — native Linux client is planned for Q3 2025."
        ),
        metadata={"category": "technical", "last_updated": "2025-02-28"},
    ),
    VectorDocument(
        id="kb-technical-002",
        text=(
            "Troubleshooting Slow Performance: If Widget Pro is running slowly, try: "
            "1) Close other applications to free RAM. "
            "2) Clear the application cache: Settings > Advanced > Clear Cache. "
            "3) Check for updates: Help > Check for Updates. "
            "4) Restart the application. "
            "If the issue persists after these steps, collect the log file (Help > Export Logs) "
            "and contact technical support."
        ),
        metadata={"category": "technical", "last_updated": "2025-03-10"},
    ),
    # ── SLA & Support Hours ───────────────────────────────────────────────
    VectorDocument(
        id="kb-support-sla-001",
        text=(
            "Support Hours & SLA: Customer support is available Monday–Friday 8am–8pm ET "
            "and Saturday 10am–6pm ET. P1 (critical) tickets are responded to within 1 hour 24/7. "
            "P2 (high) within 4 business hours. P3 (normal) within 1 business day. "
            "Enterprise customers receive a dedicated support channel with 24/7 coverage. "
            "Expected resolution times: P1 4h, P2 24h, P3 5 business days."
        ),
        metadata={"category": "policy", "last_updated": "2025-01-10"},
    ),
]


async def main(clear: bool = False) -> None:
    store = VectorStore(namespace="knowledge_base")

    if clear:
        # Remove index files and start fresh
        from pathlib import Path
        for path in [
            Path("data/embeddings/knowledge_base.npz"),
            Path("data/embeddings/knowledge_base_meta.json"),
        ]:
            if path.exists():
                path.unlink()
                print(f"Removed {path}")
        store = VectorStore(namespace="knowledge_base")  # re-init clean

    print(f"Seeding {len(DOCUMENTS)} documents into knowledge base...")
    await store.add(DOCUMENTS)
    count = store.count()
    print(f"\nDone. Total documents in store: {count}")
    print("\nCategories seeded:")
    from collections import Counter
    cats = Counter(d.metadata.get("category", "unknown") for d in DOCUMENTS)
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<20} {n} doc(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the knowledge base.")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing index before seeding")
    args = parser.parse_args()
    asyncio.run(main(clear=args.clear))
