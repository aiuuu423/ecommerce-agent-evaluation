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
    (current_gmv - previous_gmv) / nullif(previous_gmv, 0) as gmv_change_rate,
    current_orders / nullif(current_visits, 0)
      - previous_orders / nullif(previous_visits, 0) as cvr_change,
    current_refunds / nullif(current_orders, 0) as refund_rate,
    missing_days
  from traffic_metrics
  join order_metrics using (product_id)
),
reasons as (
  select
    *,
    case
      when missing_days > 0 then 'data_quality_review'
      when refund_rate >= 0.12 then 'refund_risk'
      when cvr_change <= -0.01 then 'conversion_decline'
      when gmv_change_rate <= -0.20 then 'gmv_decline'
    end as watch_reason
  from metrics
)
select
  product_id,
  watch_reason,
  gmv_change_rate,
  cvr_change,
  refund_rate,
  missing_days
from reasons
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
