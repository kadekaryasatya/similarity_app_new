from django.contrib import admin
from .models import Document,Dokumen, Komponen, DokumenKomponen, KeterkaitanDokumen, KeterkaitanKomponen

admin.site.register(Dokumen)
admin.site.register(Komponen)
admin.site.register(DokumenKomponen)
admin.site.register(KeterkaitanDokumen)
admin.site.register(KeterkaitanKomponen)
admin.site.register(Document)
