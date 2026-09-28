with bounds as (
  select max(date) as max_date
  from traffic
),
traffic_periods as (
  select
    product_id,
    case
      when date between max_date - interval 29 day and max_date then 'current'
      when date between max_date - interval 59 day and max_date - interval 30 day
        then 'previous'
    end as period,
    sum(visits) as visits
  from traffic, bounds
  where date between max_date - interval 59 day and max_date
    and not is_missing
  group by product_id, period
),
order_periods as (
  select
    product_id,
    case
      when order_date between max_date - interval 29 day and max_date then 'current'
      when order_date between max_date - interval 59 day and max_date - interval 30 day
        then 'previous'
    end as period,
    count(distinct order_id) as orders
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by product_id, period
),
combined as (
  select
    traffic_periods.product_id,
    traffic_periods.period,
    order_periods.orders / nullif(traffic_periods.visits, 0) as cvr
  from traffic_periods
  join order_periods using (product_id, period)
)
select
  product_id,
  max(case when period = 'current' then cvr end) as current_cvr,
  max(case when period = 'previous' then cvr end) as previous_cvr,
  current_cvr - previous_cvr as cvr_change
from combined
group by product_id
order by cvr_change asc nulls last, product_id
limit 10;
