# Dataset Catalog

| ID | Dataset | Domain | Unit | Peran |
|---|---|---|---|---|
| D01 | Indeks Pemerataan Guru 2025 | CAP | prov/kab-kota | sinyal pemerataan SDM |
| D02 | Jumlah GTK 2025 | CAP | regional | konteks kapasitas SDM |
| D03 | Peserta Didik Berkebutuhan Khusus | INC | regional | konteks kebutuhan layanan |
| D04 | Jumlah Satuan Pendidikan Inklusif 2025 | INC | regional | coverage layanan |
| D05 | Sarpras pendukung pendidikan inklusif 2025 | INC | regional | kesiapan layanan |
| D06 | Sekolah dengan internet untuk pengajaran 2025 | DIG | regional | dukungan digital |
| D07 | Persentase Penduduk Miskin/P0 2025 | SOC | kab/kota | konteks kerentanan |
| D08 | Master wilayah Indonesia | JOIN | nasional | harmonisasi kode wilayah |
| D09 | Dokumen kebijakan Sekolah Rakyat | POL | nasional | evidence/rule |
| D10 | Regulasi pendidikan inklusif | POL | nasional | evidence/rule |

Sumber primer yang diprioritaskan:
- repository dataset Kemendikdasmen yang dikelola pengusul;
- BPS untuk statistik sosial-ekonomi;
- JDIH Kemensos/Kemendikdasmen/BPK untuk regulasi;
- repository master wilayah cahyadsn/wilayah untuk join geografis.

Exact filename, column, denominator, coverage dan year wajib diverifikasi oleh ingestion pipeline sebelum masuk scoring.
