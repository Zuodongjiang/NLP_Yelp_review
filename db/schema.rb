# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 20181030051209) do

  # These are extensions that must be enabled in order to support this database
  enable_extension "plpgsql"
  enable_extension "postgis"

  create_table "something", primary_key: "output_id", id: :integer, default: nil, force: :cascade do |t|
    t.integer "zip"
    t.text "city"
    t.float "latitude"
    t.float "longitude"
    t.text "category"
    t.float "pop_score"
    t.float "pred_star"
    t.float "checkin_rank"
    t.json "pos_rev"
    t.json "neg_rev"
  end

end
