"""Task slice for the WebArena Verified v1 subset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebArenaTaskSpec:
    task_id: str
    site: str
    start_url: str
    expected_slug: str
    instruction: str


V1_TASKS = [
    WebArenaTaskSpec(
        task_id="search_product_specs",
        site="classifieds",
        start_url="https://classifieds.example/search",
        expected_slug="wireless-headphones",
        instruction="Search the site and open the wireless headphones listing.",
    ),
    WebArenaTaskSpec(
        task_id="navigate_support_doc",
        site="support",
        start_url="https://support.example/docs",
        expected_slug="return-policy",
        instruction="Find the return policy page from the documentation site.",
    ),
    WebArenaTaskSpec(
        task_id="filter_forum_answer",
        site="forum",
        start_url="https://forum.example/questions",
        expected_slug="gpu-setup-guide",
        instruction="Open the post that explains the GPU setup guide.",
    ),
    WebArenaTaskSpec(
        task_id="compare_product_variants",
        site="catalog",
        start_url="https://catalog.example/products",
        expected_slug="noise-cancelling-headphones",
        instruction="Compare the product variants and open the noise-cancelling headphones page.",
    ),
    WebArenaTaskSpec(
        task_id="locate_shipping_faq",
        site="support",
        start_url="https://support.example/docs",
        expected_slug="shipping-faq",
        instruction="Locate the shipping FAQ page from the documentation site.",
    ),
    WebArenaTaskSpec(
        task_id="find_forum_troubleshooting_post",
        site="forum",
        start_url="https://forum.example/questions",
        expected_slug="network-troubleshooting",
        instruction="Open the forum post that explains network troubleshooting.",
    ),
    WebArenaTaskSpec(
        task_id="open_discount_monitor_listing",
        site="classifieds",
        start_url="https://classifieds.example/search",
        expected_slug="4k-monitor-deal",
        instruction="Find and open the discounted 4K monitor listing.",
    ),
    WebArenaTaskSpec(
        task_id="compare_keyboard_variants",
        site="catalog",
        start_url="https://catalog.example/products",
        expected_slug="mechanical-keyboard-pro",
        instruction="Compare the keyboard variants and open the mechanical keyboard pro page.",
    ),
    WebArenaTaskSpec(
        task_id="locate_membership_pricing_page",
        site="support",
        start_url="https://support.example/docs",
        expected_slug="membership-pricing",
        instruction="Locate the membership pricing page from the documentation site.",
    ),
    WebArenaTaskSpec(
        task_id="find_forum_account_recovery_post",
        site="forum",
        start_url="https://forum.example/questions",
        expected_slug="account-recovery",
        instruction="Open the forum post that explains account recovery.",
    ),
    WebArenaTaskSpec(
        task_id="open_refurbished_laptop_listing",
        site="classifieds",
        start_url="https://classifieds.example/search",
        expected_slug="refurbished-laptop-deal",
        instruction="Find and open the refurbished laptop listing.",
    ),
    WebArenaTaskSpec(
        task_id="compare_router_variants",
        site="catalog",
        start_url="https://catalog.example/products",
        expected_slug="wifi-router-max",
        instruction="Compare the router variants and open the wifi router max page.",
    ),
]

TASKS_BY_ID = {task.task_id: task for task in V1_TASKS}


def get_task(task_id: str) -> WebArenaTaskSpec:
    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown WebArena Verified task_id: {task_id}") from exc


def default_task_ids() -> list[str]:
    return [task.task_id for task in V1_TASKS]
