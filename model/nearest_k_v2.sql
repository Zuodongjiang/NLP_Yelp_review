-- FUNCTION: public.nearest_k_restaurant(double precision, double precision, integer)

-- DROP FUNCTION public.nearest_k_restaurant(double precision, double precision, integer);

CREATE OR REPLACE FUNCTION public.nearest_k_restaurant_v2(
	longitude_in double precision,
	latitude_in double precision,
	k_in integer,
	category_in text)
    RETURNS TABLE(business_id_out character varying) 
    LANGUAGE 'plpgsql'

    COST 100
    VOLATILE 
    ROWS 1000
AS $BODY$BEGIN
 RETURN QUERY SELECT business_id
FROM business
where 1 = 1
AND is_us = 1
AND is_restaurant = 1
AND categories ilike '%'  || category_in || '%'
ORDER BY st_Transform(geom, 3857) <-> st_Transform(st_SetSrid(st_MakePoint(longitude_in, latitude_in), 4326), 3857) limit k_in;
END;$BODY$;

ALTER FUNCTION public.nearest_k_restaurant_v2(double precision, double precision, integer, text)
    OWNER TO xchen668;

COMMENT ON FUNCTION public.nearest_k_restaurant_v2(double precision, double precision, integer, text)
    IS 'Find nearest k restaurant v2';
