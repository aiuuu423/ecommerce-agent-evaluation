with bounds as (
  select
    max(date) as as_of_date,
    cast(max(date) - interval 29 day as date) as current_start,
    max(date) as current_end,
    cast(max(date) - interval 59 day as date) as previous_start,
    cast(max(date) - interval 30 day as date) as previous_end
  from traffic
),
traffic_metrics as (
  select
    products.product_id,
    count(traffic.date) filter (
      where traffic.date between current_start and current_end
        and not traffic.is_missing
    ) as current_observed_days,
    count(traffic.date) filter (
      where traffic.date between previous_start and previous_end
        and not traffic.is_missing
    ) as previous_observed_days,
    coalesce(sum(traffic.visits) filter (
      where traffic.date between current_start and current_end
        and not traffic.is_missing
    ), 0) as current_visits,
    coalesce(sum(traffic.visits) filter (
      where traffic.date between previous_start and previous_end
        and not traffic.is_missing
    ), 0) as previous_visits
  from products
  cross join bounds
  left join traffic
    on traffic.product_id = products.product_id
    and traffic.date between previous_start and current_end
  group by products.product_id
),
order_metrics as (
  select
    products.product_id,
    count(distinct orders.order_id) filter (
      where orders.order_date between current_start and current_end
    ) as current_orders,
    count(distinct orders.order_id) filter (
      where orders.order_date between previous_start and previous_end
    ) as previous_orders,
    coalesce(sum(orders.revenue) filter (
      where orders.order_date between current_start and current_end
    ), 0) as current_gmv,
    coalesce(sum(orders.revenue) filter (
      where orders.order_date between previous_start and previous_end
    ), 0) as previous_gmv,
    count(distinct orders.order_id) filter (
      where orders.order_date between current_start and current_end
        and orders.is_refund
    ) as current_refunds
  from products
  cross join bounds
  left join orders
    on orders.product_id = products.product_id
    and orders.order_date between previous_start and current_end
  group by products.product_id
),
metrics as (
  select
    traffic_metrics.product_id,
    (current_gmv - previous_gmv) / nullif(previous_gmv, 0) as gmv_change_rate,
    case
      when current_observed_days = 30 and previous_observed_days = 30
        then current_orders / nullif(current_visits, 0)
          - previous_orders / nullif(previous_visits, 0)
    end as cvr_change,
    current_refunds / nullif(current_orders, 0) as refund_rate,
    current_observed_days,
    previous_observed_days
  from traffic_metrics
  join order_metrics using (product_id)
),
reasons as (
  select
    *,
    case
      when current_observed_days < 30 or previous_observed_days < 30
        then 'data_quality_review'
      when refund_rate >= 0.12 then 'refund_risk'
      when cvr_change <= -0.01 then 'conversion_decline'
      when gmv_change_rate <= -0.20 then 'gmv_decline'
    end as watch_reason
  from metrics
)
select
  as_of_date,
  current_start,
  current_end,
  previous_start,
  previous_end,
  product_id,
  watch_reason,
  gmv_change_rate,
  cvr_change,
  refund_rate,
  current_observed_days,
  previous_observed_days
from reasons, bounds
where watch_reason is not null
order by
  case watch_reason
    when 'data_quality_review' then 1
    when 'refund_risk' then 2
    when 'conversion_decline' then 3
    else 4
  end,
  product_id
limit 10;
