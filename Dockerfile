# Use the python image as the base image
FROM python:3.7-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose port 5555 for the Flask app
EXPOSE 5555

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
