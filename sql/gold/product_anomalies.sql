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
    (current_visits - previous_visits) / nullif(previous_visits, 0)
      as traffic_change_rate,
    case when current_observed_days = 30
      then current_orders / nullif(current_visits, 0) end as current_cvr,
    case when previous_observed_days = 30
      then previous_orders / nullif(previous_visits, 0) end as previous_cvr,
    current_refunds / nullif(current_orders, 0) as current_refund_rate,
    (current_gmv - previous_gmv) / nullif(previous_gmv, 0) as gmv_change_rate,
    (
      current_gmv / nullif(current_orders, 0)
      - previous_gmv / nullif(previous_orders, 0)
    ) / nullif(previous_gmv / nullif(previous_orders, 0), 0) as aov_change_rate,
    current_observed_days,
    previous_observed_days
  from traffic_metrics
  join order_metrics using (product_id)
),
labeled as (
  select
    *,
    case
      when current_observed_days < 30 or previous_observed_days < 30
        then 'missing_traffic'
      when current_refund_rate >= 0.12 then 'high_refund'
      when traffic_change_rate <= -0.30 and gmv_change_rate <= -0.30
        then 'multi_factor_drop'
      when traffic_change_rate <= -0.30 then 'traffic_drop'
      when current_cvr - previous_cvr <= -0.01 then 'conversion_drop'
      when gmv_change_rate <= -0.30 and aov_change_rate <= -0.30
        then 'sales_drop'
    end as anomaly_type
  from metrics
)
select
  as_of_date,
  current_start,
  current_end,
  previous_start,
  previous_end,
  product_id,
  anomaly_type,
  traffic_change_rate,
  current_cvr,
  previous_cvr,
  current_refund_rate,
  gmv_change_rate,
  aov_change_rate,
  current_observed_days,
  previous_observed_days
from labeled, bounds
where anomaly_type is not null
order by product_id;
