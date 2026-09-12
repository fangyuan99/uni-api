from uni_api.routing.core import (  # noqa: F401
    ProviderAttempt,
    RoutingPlan,
    _call_provider_resolver,
    compute_retry_count,
    get_matching_providers,
    get_provider_list,
    get_provider_rules,
    get_right_order_providers,
    is_virtual_api_key_channel,
    select_provider_api_key_raw,
    virtual_api_key_index,
)


class RoutingPlanner:
    async def plan(self, *args, **kwargs) -> RoutingPlan:
        return await RoutingPlan.create(*args, **kwargs)
