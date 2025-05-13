from django.urls import path
from .views import upload_zip, train_model, training_logs, training_status, get_train_results

urlpatterns = [
    path("upload_zip/", upload_zip),
    path("train/", train_model),
    path("logs/", training_logs),
    path("status/", training_status),
    path("train_results/", get_train_results),
]