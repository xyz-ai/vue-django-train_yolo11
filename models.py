from django.db import models

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

class UploadedDataset(models.Model):
    folder_name = models.CharField(max_length=256)  
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.folder_name
