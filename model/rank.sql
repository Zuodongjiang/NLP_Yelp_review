GRANT ALL ON public.checkin TO PUBLIC;

select * from review limit 1; 
select r.business_id, avg(CAST(r.stars as float)) FROM
(select public.nearest_k_restaurant(-112.0685, 33.4528, 1) as business_id) a
inner join review r
on a.business_id = r.business_id
group by r.business_id;

DROP TABLE IF EXISTS predicted_star;
CREATE TABLE predicted_star AS
select r.business_id, avg(CAST(r.stars as float)) as pred_star, count(r.stars) as checkin, b.city
FROM business b
inner join review r
on b.business_id = r.business_id
where b.is_us = 1
and b.is_restaurant = 1
and b.city in ('Cleveland', 'Phoenix')
group by b.city, r.business_id;

DROP TABLE IF EXISTS predicted_star_rank;
CREATE TABLE predicted_star_rank AS
SELECT business_id, pred_star, checkin,
	(PERCENT_RANK() 
    OVER (PARTITION BY city ORDER BY checkin))/20*5 AS perc_rank,
   city
FROM predicted_star;
							   
select sum(perc_rank) from predicted_star_rank r
inner join business b
on r.business_id = b.business_id
where b.city = 'Cleveland' limit 10;
											   
select sum(perc_rank)*2 from predicted_star_rank r
inner join business b
on r.business_id = b.business_id
where b.city = 'Phoenix' limit 10;
-- select count(*) from predicted_star where perc_rank = 0 limit 10;
 
select avg(r.pred_star) + sum(r.perc_rank) AS popularity FROM
(select public.nearest_k_restaurant(-112.0685, 33.4528, 20) as business_id) a
inner join predicted_star_rank r
on a.business_id = r.business_id;

33.5079722,-112.10129119999999
select avg(r.pred_star) + sum(r.perc_rank) AS popularity FROM
(select public.nearest_k_restaurant(-112.10129119999999, 33.5079722, 20) as business_id) a
inner join predicted_star_rank r
on a.business_id = r.business_id;
		  
select avg(r.pred_star) + sum(r.perc_rank) AS popularity FROM
(select public.nearest_k_restaurant(-112.10, 33.4879722, 20) as business_id) a
inner join predicted_star_rank r
on a.business_id = r.business_id;
							   
--select * from predicted_star_rank limit 100;
						  
--select count(distinct(business_id)) from predicted_star_rank where city = 'Cleveland';
--select count(*) from predicted_star_rank where city = 'Phoenix';