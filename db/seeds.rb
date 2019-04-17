require 'csv'

csv_text = File.read(Rails.root.join('lib', 'seeds', '6242.csv'))
csv = CSV.parse(csv_text, :headers => true, :encoding => 'ISO-8859-1')
csv.each do |row|
  t = Something.new
  t.output_id = row['output_id']
	t.zip = row['zip']
  t.city = row['city']
  t.latitude = row['latitude']
  t.longitude = row['longitude']
  t.category = row['category']
  t.pop_score = row['pop_score']
  t.pred_star = row['pred_star']
  t.checkin_rank = row['checkin_rank']
  t.pos_rev = JSON.parse(row['pos_rev'])
  t.neg_rev	 = JSON.parse(row['neg_rev'])
  t.save
  puts "#{t.output_id}, #{t.city} saved"
end

puts "There are now #{Something.count} rows in the something table"

# # This file should contain all the record creation needed to seed the database with its default values.
# # The data can then be loaded with the rails db:seed command (or created alongside the database with db:setup).
# #
# # Examples:
# #
# #   movies = Movie.create([{ name: 'Star Wars' }, { name: 'Lord of the Rings' }])
# #   Character.create(name: 'Luke', movie: movies.first)
#
# # (output_id int PRIMARY KEY,
# # zip int,
# # city text,
# # latitude float,
# # longitude float,
# # category text,
# # pop_score float,
# # pos_rev text,
# # neg_rev text
# # );
# #
# # categories = ['American', 'Breakfast', 'Chinese', 'Italian', 'Nightlife', 'Pizza']
# #
# # cities_array = ["Las Vegas", "Phoenix", "Charlotte", "Pittsburgh", "Scottsdale", "Cleveland", "Mesa", "Madison", "Tempe", "Henderson", "Houston"]
# #
# # i = 0
# #
# # cities_array.each do |city|
# #
# # 	Geocoder.search(city).each do |c|
# #
# # 		next unless (c.postal_code && c.postal_code.to_i != 0 && c.latitude && c.longitude)
# # 		i += 1
# #
# # 		hash = {}
# # 		hash[:output_id] = i
# # 		hash[:city] = city
# # 		hash[:zip] = c.postal_code
# # 		hash[:latitude] = c.latitude
# # 		hash[:longitude] = c.longitude
# # 		hash[:category] = categories.sample
# # 		hash[:pop_score] = rand(1..10)
# # 		hash[:pred_star] = rand(1..100)
# # 		hash[:checkin_rank] = rand(1..100)
# # 		hash[:pos_rev] = {"service": 0.494, "table": 0.217, "don": 0.215, "restaurant": 0.197, "try": 0.186, "drinks": 0.173, "food": 0.161, "place": 0.131, "just": 0.12, "ramen": 0.111}
# # 		hash[:neg_rev] = {"service": 0.494, "table": 0.217, "don": 0.215, "restaurant": 0.197, "try": 0.186, "drinks": 0.173, "food": 0.161, "place": 0.131, "just": 0.12, "ramen": 0.111}
# #
# # 		Something.create(
# # 			hash
# # 		)
# #
# # 	end
# #
# #
# # end
# seeds = [
# 	{
# 		output_id: 101,
# 		zip: 85004,
# 		city: "Phoenix",
# 		latitude: 33.4528292,
# 		longitude: -112.0685027,
# 		category: "American",
# 		pop_score: 6.36745315820634,
# 		pred_star: 3.30550683199477,
# 		checkin_rank: 3.06194632621157,
# 		neg_rev: {"service": 0.494, "table": 0.217, "don": 0.215, "restaurant": 0.197, "try": 0.186, "drinks": 0.173, "food": 0.161, "place": 0.131, "just": 0.12, "ramen": 0.111}
# 	},
# 	{
# 		output_id: 102,
# 		zip: 85042,
# 		city: "Phoenix",
# 		latitude: 33.3762931,
# 		longitude: -112.0357137,
# 		category: "Chinese",
# 		pop_score: 5.58572392029118,
# 		pred_star: 3.05341542628388,
# 		checkin_rank: 2.5323084940073,
# 		neg_rev: {"service": 0.461, "ordered": 0.276, "waitress": 0.217, "said": 0.217, "eating": 0.184, "took": 0.165, "bone": 0.154, "eat": 0.139, "zero": 0.13, "order": 0.128}
# 	},
# 	{
# 		output_id: 103,
# 		zip: 85013,
# 		city: "Phoenix",
# 		latitude: 33.3762931,
# 		longitude: -112.0863874,
# 		category: "Nightlife",
# 		pop_score: 4.67621997345615,
# 		pred_star: 2.95585780566042,
# 		checkin_rank: 1.72036216779573,
# 		neg_rev: {"service": 0.354, "tacos": 0.268, "restaurant": 0.268, "didn": 0.221, "don": 0.215, "people": 0.19, "table": 0.179, "times": 0.144, "food": 0.142, "drinks": 0.129}
# 	},
# 	{
# 		output_id: 105,
# 		zip: 85012,
# 		city: "Phoenix",
# 		latitude: 33.5114334,
# 		longitude: -112.0685027,
# 		category: "Italian",
# 		pop_score: 7.16130236982622,
# 		pred_star: 3.63511685653805,
# 		checkin_rank: 3.52618551328817,
# 		neg_rev: {"table": 0.444, "restaurant": 0.378, "food": 0.187, "just": 0.18, "coffee": 0.159, "came": 0.133, "took": 0.124, "party": 0.119, "like": 0.115, "try": 0.113}
# 	},
# 	{
# 		output_id: 106,
# 		zip: 44113,
# 		city: "Cleveland",
# 		latitude: 41.4857101,
# 		longitude: -81.6966306,
# 		category: "American",
# 		pop_score: 6.02530012083413,
# 		pred_star: 3.79012016317711,
# 		checkin_rank: 2.23517995765702,
# 		neg_rev: {"service": 0.369, "ordered": 0.341, "didn": 0.263, "table": 0.251, "drinks": 0.185, "experience": 0.174, "people": 0.168, "restaurant": 0.16, "don": 0.157, "food": 0.144}
# 	},
# 	{
# 		output_id: 107,
# 		zip: 44114,
# 		city: "Cleveland",
# 		latitude: 41.5139193,
# 		longitude: -81.6747295,
# 		category: "Chinese",
# 		pop_score: 6.81292633633982,
# 		pred_star: 3.85685717614222,
# 		checkin_rank: 2.9560691601976,
# 		neg_rev: {"restaurant": 0.264, "service": 0.263, "chinese": 0.232, "sauce": 0.228, "taste": 0.21, "pretty": 0.187, "fried": 0.183, "dishes": 0.183, "noodles": 0.176, "tasted": 0.172}
# 	},
# 	{
# 		output_id: 108,
# 		zip: 44106,
# 		city: "Cleveland",
# 		latitude: 41.5091257,
# 		longitude: -81.60898735,
# 		category: "Chinese",
# 		pop_score: 6.32441738814577,
# 		pred_star: 3.70462204587337,
# 		checkin_rank: 2.61979534227241,
# 		neg_rev: {"restaurant": 0.396, "service": 0.262, "sauce": 0.258, "food": 0.209, "buffet": 0.205, "taste": 0.184, "dishes": 0.17, "tasted": 0.161, "fried": 0.149, "chinese": 0.138}
# 	},
# 	{
# 		output_id: 109,
# 		zip: 44106,
# 		city: "Cleveland",
# 		latitude: 41.5091257,
# 		longitude: -81.60898735,
# 		category: "Pizza",
# 		pop_score: 6.32441738814577,
# 		pred_star: 3.70462204587337,
# 		checkin_rank: 2.61979534227241,
# 		neg_rev: {"little": 0.37, "italy": 0.277, "service": 0.267, "sauce": 0.246, "don": 0.22, "pizza": 0.207, "cheese": 0.199, "restaurant": 0.165, "people": 0.164, "minutes": 0.155}
# 	},
# 	{
# 		output_id: 110,
# 		zip: 44114,
# 		city: "Cleveland",
# 		latitude: 41.5139193,
# 		longitude: -81.6747295,
# 		category: "Breakfast",
# 		pop_score: 6.81292633633982,
# 		pred_star: 3.85685717614222,
# 		checkin_rank: 2.9560691601976,
# 		neg_rev: {"service": 0.411, "minutes": 0.334, "coffee": 0.267, "manager": 0.19, "restaurant": 0.177, "busy": 0.17, "really": 0.142, "friendly": 0.125, "convention": 0.122, "people": 0.116}
# 	}
# ]
#
# seeds.each do |seed|
# 	Something.create(
# 		seed
# 	)
# end
