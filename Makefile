build:
	poetry export --without-hashes -f requirements.txt -o requirements.txt
	docker build -t ml-course .

run-env:
	docker run --rm -v "$(shell pwd):/app" --name ml-course-c -it ml-course bash
