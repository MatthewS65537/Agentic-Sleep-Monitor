from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

# Create your views here.

def dataview(request):
    return render(request, 'dataview/dataview.html')

@require_http_methods(["GET"])
def get_json_data(request):
    # This is a placeholder for your actual data source
    sample_data = {
        "type": "sample_type",
        "data": {
            "key1": "value1",
            "key2": "value2",
            "nested": {
                "nestedKey": "nestedValue"
            }
        }
    }
    return JsonResponse(sample_data)
