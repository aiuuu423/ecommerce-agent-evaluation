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
        when order_date between max_date - interval 29 day and max_date then revenue
        else 0
      end
    ) as current_gmv,
    sum(
      case
        when order_date between max_date - interval 59 day and max_date - interval 30 day
          then revenue
        else 0
      end
    ) as previous_gmv,
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
    current_orders / nullif(current_visits, 0) as current_cvr,
    previous_orders / nullif(previous_visits, 0) as previous_cvr,
    current_refunds / nullif(current_orders, 0) as current_refund_rate,
    (current_gmv - previous_gmv) / nullif(previous_gmv, 0) as gmv_change_rate,
    missing_days
  from traffic_metrics
  join order_metrics using (product_id)
),
labeled as (
  select
    *,
    case
      when missing_days > 0 then 'missing_traffic'
      when current_refund_rate >= 0.12 then 'high_refund'
      when traffic_change_rate <= -0.30 and gmv_change_rate <= -0.30
        then 'multi_factor_drop'
      when traffic_change_rate <= -0.30 then 'traffic_drop'
      when current_cvr - previous_cvr <= -0.01 then 'conversion_drop'
      when gmv_change_rate <= -0.30 then 'sales_drop'
    end as anomaly_type
  from metrics
)
select
  product_id,
  anomaly_type,
  traffic_change_rate,
  current_cvr,
  previous_cvr,
  current_refund_rate,
  gmv_change_rate,
  missing_days
from labeled
where anomaly_type is not null
order by product_id;
