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
priorities as (
  select
    product_id,
    current_observed_days,
    previous_observed_days,
    case
      when current_observed_days < 30 or previous_observed_days < 30
        then 'repair_data_quality'
      when refund_rate >= 0.12 then 'investigate_refunds'
      when cvr_change <= -0.01 then 'recover_conversion'
      when traffic_change_rate <= -0.20 then 'recover_traffic'
    end as priority_reason,
    case
      when current_observed_days < 30 or previous_observed_days < 30
        then 'observed_days'
      when refund_rate >= 0.12 then 'refund_rate'
      when cvr_change <= -0.01 then 'cvr_change'
      when traffic_change_rate <= -0.20 then 'traffic_change_rate'
    end as evidence_metric,
    case
      when current_observed_days < 30 or previous_observed_days < 30
        then cast(least(current_observed_days, previous_observed_days) as double)
      when refund_rate >= 0.12 then refund_rate
      when cvr_change <= -0.01 then cvr_change
      when traffic_change_rate <= -0.20 then traffic_change_rate
    end as evidence_value
  from metrics
)
select
  as_of_date,
  current_start,
  current_end,
  previous_start,
  previous_end,
  product_id,
  priority_reason,
  evidence_metric,
  evidence_value,
  current_observed_days,
  previous_observed_days
from priorities, bounds
where priority_reason is not null
order by
  case priority_reason
    when 'repair_data_quality' then 1
    when 'investigate_refunds' then 2
    when 'recover_conversion' then 3
    else 4
  end,
  abs(evidence_value) desc,
  product_id
limit 10;
