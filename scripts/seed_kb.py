"""
Seed the knowledge base vector store with initial documents.

Run: python scripts/seed_kb.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.vector_store import VectorStore, VectorDocument

DOCUMENTS = [
    VectorDocument(
        id="kb-001",
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
        id="kb-002",
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
        id="kb-003",
        text=(
            "Warranty Information: All hardware products carry a 2-year manufacturer warranty against defects. "
            "Accidental damage is not covered under warranty. "
            "To file a warranty claim, provide proof of purchase and a description of the defect. "
            "Warranty replacements are shipped within 5 business days of approval. "
            "Extended warranty plans are available for purchase within 90 days of original purchase."
        ),
        metadata={"category": "warranty", "last_updated": "2024-11-20"},
    ),
    VectorDocument(
        id="kb-004",
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
        id="kb-005",
        text=(
            "Privacy Policy: We collect personal information only as necessary to fulfill orders and provide support. "
            "We never sell customer data to third parties. "
            "You may request a copy of your data or request deletion at any time by emailing privacy@company.com. "
            "We are GDPR compliant and CCPA compliant. "
            "Data is retained for 7 years for financial records, 2 years for support interactions."
        ),
        metadata={"category": "policy", "last_updated": "2025-01-01"},
    ),
]


async def main():
    store = VectorStore(namespace="knowledge_base")
    print(f"Seeding {len(DOCUMENTS)} documents into knowledge base...")
    await store.add(DOCUMENTS)
    print(f"Done. Total documents: {store.count()}")


if __name__ == "__main__":
    asyncio.run(main())
