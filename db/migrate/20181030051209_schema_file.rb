class SchemaFile < ActiveRecord::Migration[5.1]
  def change

    execute <<-SQL

    --CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

  --  CREATE TABLE response
  --    (
  --      response_id SERIAL PRIMARY KEY,
    --      name TEXT,
  --      response_uuid uuid NOT NULL DEFAULT uuid_generate_v1()
  --    );

    DROP TABLE IF EXISTS something;
    DROP TABLE IF EXISTS choi_test;

    CREATE TABLE something
    (
      output_id int PRIMARY KEY,
      zip int,
      city text,
      latitude float,
      longitude float,
      category text,
      pop_score float,
      pred_star float,
      checkin_rank float,
      pos_rev json,
      neg_rev json
    );

    SQL
  end
end
