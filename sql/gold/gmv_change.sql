with bounds as (
  select max(order_date) as max_date
  from orders
),
periods as (
  select
    case
      when order_date between max_date - interval 29 day and max_date then 'current'
      when order_date between max_date - interval 59 day and max_date - interval 30 day
        then 'previous'
    end as period,
    sum(revenue) as gmv
  from orders, bounds
  where order_date between max_date - interval 59 day and max_date
  group by period
)
select
  max(case when period = 'current' then gmv end) as current_gmv,
  max(case when period = 'previous' then gmv end) as previous_gmv,
  (current_gmv - previous_gmv) / nullif(previous_gmv, 0) as gmv_change_rate
from periods
order by current_gmv, previous_gmv;
