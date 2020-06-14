from django.urls import path
from . import views
from django.conf.urls import url, include
urlpatterns = [
    path('upload',views.upload,name = 'upload'),
    path('scatter',views.scatter,name = 'scatter'),
    path('scatterplot',views.scatterplot,name = 'scatterplot'),
    path('box',views.box,name = 'box'),
    path('boxplot',views.boxplot,name = 'boxplot'),
    path('violin',views.violin,name = 'violin'),
    path('violinplot',views.violinplot,name = 'violinplot'),
    path('dist',views.dist,name = 'dist'),
    path('distplot',views.distplot,name = 'distplot'),
    path('swarm',views.swarm,name = 'swarm'),
    path('swarmplot',views.swarmplot,name = 'swarmplot'),
    path('features',views.features, name='features'),
    path('classifier',views.classifier,name = 'classifier'),
    path('',views.home,name='home'),
    path('aboutfeatures',views.aboutfeatures,name='aboutfeatures'),
    path('homereturn',views.homereturn,name='homereturn'),
    path('AboutClassifier',views.aboutclassifier, name='aboutclassifer'),
]
