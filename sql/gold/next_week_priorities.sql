with bounds as (
  select max(date) as max_date
  from traffic
),
traffic_metrics as (
  select
    product_id,
    sum(case when date between max_date - interval 29 day and max_date then visits end)
      as current_visits,
    sum(
      case
        when date between max_date - interval 59 day and max_date - interval 30 day
          then visits
      end
    ) as previous_visits,
    sum(
      case
        when date between max_date - interval 29 day and max_date and is_missing
          then 1
        else 0
      end
    ) as missing_days
  from traffic, bounds
  where date between max_date - interval 59 day and max_date
  group by product_id
),
order_metrics as (
  select
    product_id,
    count(
      distinct case
        when order_date between max_date - interval 29 day and max_date then order_id
      end
    ) as current_orders,
    count(
      distinct case
        when order_date between max_date - interval 59 day and max_date - interval 30 day
          then order_id
      end
    ) as previous_orders,
    sum(
      case
        when order_date between max_date - interval 29 day and max_date and is_refund
          then 1
        else 0
      end
    ) as current_refunds
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by product_id
),
metrics as (
  select
    traffic_metrics.product_id,
    (current_visits - previous_visits) / nullif(previous_visits, 0)
      as traffic_change_rate,
    current_orders / nullif(current_visits, 0)
      - previous_orders / nullif(previous_visits, 0) as cvr_change,
    current_refunds / nullif(current_orders, 0) as refund_rate,
    missing_days
  from traffic_metrics
  join order_metrics using (product_id)
),
priorities as (
  select
    product_id,
    case
      when missing_days > 0 then 'repair_data_quality'
      when refund_rate >= 0.12 then 'investigate_refunds'
      when cvr_change <= -0.01 then 'recover_conversion'
      when traffic_change_rate <= -0.20 then 'recover_traffic'
    end as priority_reason,
    case
      when missing_days > 0 then 'missing_days'
      when refund_rate >= 0.12 then 'refund_rate'
      when cvr_change <= -0.01 then 'cvr_change'
      when traffic_change_rate <= -0.20 then 'traffic_change_rate'
    end as evidence_metric,
    case
      when missing_days > 0 then cast(missing_days as double)
      when refund_rate >= 0.12 then refund_rate
      when cvr_change <= -0.01 then cvr_change
      when traffic_change_rate <= -0.20 then traffic_change_rate
    end as evidence_value
  from metrics
)
select product_id, priority_reason, evidence_metric, evidence_value
from priorities
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
