"""
URL configuration for similarity_app project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# similarity_docs_app/similarity_docs_app/urls.py

from django.contrib import admin
from django.urls import path
from documents import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('', views.home, name='home'),
    path('', views.home_new, name='home'),
    # path('document/<int:doc_id>/', views.view_document, name='view_document'),
    path('document/<int:doc_id>/', views.view_document_new, name='view_document'),
    # path('add/', views.add_document, name='add_document'),
    path('add/', views.add_document_new, name='add_document'),
    # path('delete/<int:doc_id>/', views.delete_document_view, name='delete_document'),  # Update nama view  
    path('delete/<int:doc_id>/', views.delete_document_view_new, name='delete_document'),  # Update nama view  
    # path('similarity/', views.similarity, name='similarity'),
    path('similarity/', views.similarity_new, name='similarity'),

    # path('similarity-graph/', views.similarity_graph, name='similarity_graph'),
    path('similarity-graph/', views.similarity_graph_new, name='similarity_graph'),

    # path('similarity_detail/<int:i>/<int:j>/', views.similarity_detail, name='similarity_detail'),
    path('similarity_detail/<int:i>/<int:j>/', views.similarity_detail_new, name='similarity_detail'),

    path('clustering/', views.clustering_new, name='clustering'),

    path('similarity_graph_detail/<int:doc_id>/', views.similarity_graph_detail, name='similarity_graph_detail'),

    # path('debug_graph/<int:doc_id>/', views.debug_similarity_graph, name='debug_graph'),

]  + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

