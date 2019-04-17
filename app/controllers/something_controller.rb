class SomethingController < ApplicationController

	def show

		if Rails.env == "development"
			unless @record
				@record = ActiveRecord::Base.connection.execute("
					SELECT * FROM something WHERE city = '#{params["id"]}' AND category = '#{params["category"]}'
				")
			end
		else
			unless @record
				@record = ActiveRecord::Base.connection.execute("
					SELECT * FROM output_revised_v2 WHERE city = '#{params["id"]}' AND category = '#{params["category"]}'
					")
			end
		end

	end

	def new
		@record = Something.new
	end

	def create

		if Rails.env == "development"
			@record = ActiveRecord::Base.connection.execute("
					SELECT * FROM something WHERE city = '#{something_params["city"]}' AND category = '#{something_params["category"]}'
				")
		else
			@record = ActiveRecord::Base.connection.execute("
					SELECT * FROM output_revised_v2
					WHERE city = '#{something_params["city"]}'
					AND category = '#{something_params["category"]}'
				")
		end

		redirect_to something_path(id: @record.first["city"], category: @record.first["category"])

	end


	private

  def something_params

    params[:something].permit(:city, :category)

  end

end
