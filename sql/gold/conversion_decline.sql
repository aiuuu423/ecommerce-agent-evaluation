with bounds as (
  select
    max(date) as as_of_date,
    cast(max(date) - interval 29 day as date) as current_start,
    max(date) as current_end,
    cast(max(date) - interval 59 day as date) as previous_start,
    cast(max(date) - interval 30 day as date) as previous_end
  from traffic
),
periods(period, period_start, period_end) as (
  select 'current', current_start, current_end from bounds
  union all
  select 'previous', previous_start, previous_end from bounds
),
product_periods as (
  select products.product_id, periods.*
  from products
  cross join periods
),
traffic_periods as (
  select
    product_periods.product_id,
    product_periods.period,
    count(traffic.date) filter (where not traffic.is_missing) as observed_days,
    coalesce(sum(traffic.visits) filter (where not traffic.is_missing), 0) as visits
  from product_periods
  left join traffic
    on traffic.product_id = product_periods.product_id
    and traffic.date between product_periods.period_start and product_periods.period_end
  group by product_periods.product_id, product_periods.period
),
order_periods as (
  select
    product_periods.product_id,
    product_periods.period,
    count(distinct orders.order_id) as orders
  from product_periods
  left join orders
    on orders.product_id = product_periods.product_id
    and orders.order_date between product_periods.period_start and product_periods.period_end
  group by product_periods.product_id, product_periods.period
),
period_metrics as (
  select
    traffic_periods.product_id,
    traffic_periods.period,
    traffic_periods.observed_days,
    traffic_periods.visits,
    order_periods.orders,
    case
      when traffic_periods.observed_days = 30
        then order_periods.orders / nullif(traffic_periods.visits, 0)
    end as cvr
  from traffic_periods
  join order_periods using (product_id, period)
),
product_metrics as (
  select
    product_id,
    max(cvr) filter (where period = 'current') as current_cvr,
    max(cvr) filter (where period = 'previous') as previous_cvr,
    max(orders) filter (where period = 'current') as current_orders,
    max(orders) filter (where period = 'previous') as previous_orders,
    max(visits) filter (where period = 'current') as current_visits,
    max(visits) filter (where period = 'previous') as previous_visits,
    max(observed_days) filter (where period = 'current') as current_observed_days,
    max(observed_days) filter (where period = 'previous') as previous_observed_days
  from period_metrics
  group by product_id
)
select
  as_of_date,
  current_start,
  current_end,
  previous_start,
  previous_end,
  product_id,
  current_cvr,
  previous_cvr,
  current_cvr - previous_cvr as cvr_change,
  current_orders,
  previous_orders,
  current_visits,
  previous_visits,
  current_observed_days,
  previous_observed_days
from product_metrics, bounds
where current_cvr - previous_cvr < 0
order by cvr_change asc, product_id
limit 10;
