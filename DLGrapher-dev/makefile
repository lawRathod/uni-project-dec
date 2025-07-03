run-inference:
	@echo "Running inference..."
	export PYTHONPATH=/home/law/project-mia/DLGrapher-dev && \
	cd src && \
	python main.py model=hybrid dataset=twitch +experiment=tablegraph_mia ++general.name=twitch_mia ++general.test_only=../checkpoints/twitch/twitch.ckpt ++general.final_model_samples_to_generate=1