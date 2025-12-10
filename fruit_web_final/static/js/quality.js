document.addEventListener('DOMContentLoaded', () => {
    console.log('수확의 정석 품질 분류 시스템 초기화 완료');

    // fileList에 스크롤 스타일 적용
    const fileList = document.getElementById('fileList');
    if (fileList) {
        fileList.style.overflowY = 'auto';
        fileList.style.maxHeight = 'calc(50vh - 200px)';
        fileList.style.minHeight = '150px';
    }

    let uploadedImages = [];
    let uploadedCSV = null;
    let processedResults = [];
    let qualityChart = null;
    let showChartLabels = false;

    if (typeof Chart === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
        script.onload = () => {
            console.log('Chart.js 로드 완료');
            initChart();
        };
        document.head.appendChild(script);
    } else {
        initChart();
    }

    function initChart() {
        const ctx = document.getElementById('qualityChart');
        if (ctx) {
            qualityChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['특', '상'],
                    datasets: [{
                        data: [0, 0],
                        backgroundColor: 'rgba(37, 99, 235, 0.3)',
                        borderColor: 'rgba(37, 99, 235, 0.5)',
                        borderWidth: 2,
                        barThickness: 80
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {
                        padding: {
                            top: 30,
                            left: 20
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                font: {
                                    size: 14,
                                    weight: 'bold'
                                }
                            }
                        },
                        y: {
                            display: false,
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1,
                                precision: 0
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            enabled: true
                        },
                        datalabels: {
                            anchor: 'end',
                            align: 'top',
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            color: '#1f2937',
                            formatter: function(value) {
                                return value;
                            }
                        }
                    }
                },
                plugins: [{
                    id: 'customDataLabels',
                    afterDatasetsDraw: function(chart) {
                        if (!showChartLabels) return;

                        const ctx = chart.ctx;
                        chart.data.datasets.forEach(function(dataset, i) {
                            const meta = chart.getDatasetMeta(i);
                            meta.data.forEach(function(bar, index) {
                                const data = dataset.data[index];

                                ctx.fillStyle = '#1f2937';
                                ctx.font = 'bold 16px sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'bottom';

                                const x = bar.x;
                                const y = bar.y - 5;

                                ctx.fillText(data, x, y);
                            });
                        });
                    }
                }]
            });
        }
    }

    function updateChart(results) {
        if (!qualityChart) return;

        const gradeCounts = { '특': 0, '상': 0 };

        results.forEach(result => {
            const grade = result.quality_grade;
            if (gradeCounts.hasOwnProperty(grade)) {
                gradeCounts[grade]++;
            }
        });

        showChartLabels = true;
        qualityChart.data.datasets[0].data = [gradeCounts['특'], gradeCounts['상']];
        qualityChart.update();
    }

    const navLinks = document.querySelectorAll('.nav-item > a');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = e.target.getAttribute('href');

            if (href && href.startsWith('#')) {
                e.preventDefault();
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                console.log('선택된 메뉴:', href);
            }
        });
    });

    const dropArea = document.getElementById('dropArea');
    const imageInput = document.getElementById('imageInput');
    const csvInput = document.getElementById('csvInput');
    const imageSelectBtn = document.getElementById('imageSelectBtn');
    const csvSelectBtn = document.getElementById('csvSelectBtn');
    const emptyState = document.getElementById('emptyState');
    const csvStatus = document.getElementById('csvStatus');
    const csvFileName = document.getElementById('csvFileName');
    const csvRemoveBtn = document.getElementById('csvRemoveBtn');

    imageSelectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.click();
    });

    csvSelectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        csvInput.click();
    });

    dropArea.addEventListener('click', (e) => {
        if (e.target === dropArea || e.target.closest('.empty-state')) {
            imageInput.click();
        }
    });

    imageInput.addEventListener('change', (e) => {
        handleImageFiles(e.target.files);
        imageInput.value = '';
    });

    csvInput.addEventListener('change', (e) => {
        handleCSVFile(e.target.files[0]);
        csvInput.value = '';
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropArea.classList.remove('dragover');
        });
    });

    dropArea.addEventListener('drop', (e) => {
        const files = Array.from(e.dataTransfer.files);
        const imageFiles = files.filter(f => f.type.startsWith('image/'));
        const csvFile = files.find(f => f.name.toLowerCase().endsWith('.csv'));

        if (imageFiles.length > 0) {
            handleImageFiles(imageFiles);
        }
        if (csvFile) {
            handleCSVFile(csvFile);
        }
    });

    function handleImageFiles(files) {
        const validFiles = Array.from(files).filter(file => {
            const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
            return validTypes.includes(file.type);
        });

        validFiles.forEach(file => {
            const fileObj = {
                id: Date.now() + Math.random(),
                file: file,
                name: file.name
            };

            uploadedImages.push(fileObj);
            createFileCard(fileObj);
        });

        updateEmptyState();
    }

    function handleCSVFile(file) {
        if (!file || !file.name.toLowerCase().endsWith('.csv')) {
            alert('CSV 파일만 업로드 가능합니다.');
            return;
        }

        uploadedCSV = file;
        csvFileName.textContent = file.name;
        csvStatus.style.display = 'block';
        updateEmptyState();
        console.log('CSV 파일 업로드:', file.name);
    }

    csvRemoveBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        uploadedCSV = null;
        csvStatus.style.display = 'none';
        updateEmptyState();
    });

    function createFileCard(fileObj) {
        const card = document.createElement('div');
        card.className = 'file-card';
        card.dataset.fileId = fileObj.id;

        const reader = new FileReader();
        reader.onload = (e) => {
            card.innerHTML = `
                <button class="delete-btn" onclick="deleteImage(${fileObj.id})">
                    <img src="/static/images/icon_delete.png" alt="삭제" width="14" height="14">
                </button>
                <img src="${e.target.result}" alt="${fileObj.name}" class="file-preview">
                <div class="file-name" title="${fileObj.name}">${fileObj.name}</div>
            `;
        };
        reader.readAsDataURL(fileObj.file);

        fileList.appendChild(card);
    }

    function updateEmptyState() {
        if (uploadedImages.length > 0 || uploadedCSV) {
            emptyState.classList.add('hidden');
        } else {
            emptyState.classList.remove('hidden');
        }
    }

    window.deleteImage = function(fileId) {
        uploadedImages = uploadedImages.filter(f => f.id !== fileId);

        const card = document.querySelector(`[data-file-id="${fileId}"]`);
        if (card) {
            card.remove();
        }

        updateEmptyState();
        console.log('파일 삭제됨. 남은 파일:', uploadedImages.length);
    };

    function initResults(totalCount) {
        const infoContent = document.getElementById('infoContent');
        infoContent.innerHTML = `
            <div class="result-section">
                <p id="progressText">총 ${totalCount}개 이미지 처리중 (0/${totalCount})</p>
                <div class="result-images" id="resultImages"></div>
            </div>
        `;

        // resultImages에 스크롤 스타일 적용
        const resultImages = document.getElementById('resultImages');
        if (resultImages) {
            resultImages.style.overflowY = 'auto';
            resultImages.style.maxHeight = 'calc(50vh - 200px)';
            resultImages.style.minHeight = '150px';
        }
    }

    function addResult(result, currentIndex, totalCount) {
        const resultImages = document.getElementById('resultImages');
        const progressText = document.getElementById('progressText');

        const resultItem = document.createElement('div');
        resultItem.className = 'result-image-item';
        resultItem.innerHTML = `
            <img src="${result.quality_image}" alt="품질 분류 결과 ${currentIndex}">
            <p>${result.filename} - ${result.quality_grade}</p>
        `;
        resultImages.appendChild(resultItem);

        progressText.textContent = `총 ${totalCount}개 이미지 처리중 (${currentIndex}/${totalCount})`;
    }

    function completeResults(successCount, totalCount) {
        const progressText = document.getElementById('progressText');
        if (successCount === 0) {
            progressText.innerHTML = `<span style="color: #ef4444;">과일을 감지하지 못했습니다 (0/${totalCount})</span>`;
        } else {
            progressText.textContent = `총 ${totalCount}개 이미지 처리 완료 (${successCount}/${totalCount})`;
        }

        updateChart(processedResults);
    }

    const runBtn = document.getElementById('runBtn');
    runBtn.addEventListener('click', async () => {
        if (uploadedImages.length === 0) {
            alert('참외 이미지를 먼저 업로드해주세요.');
            return;
        }

        if (!uploadedCSV) {
            alert('CSV 파일을 먼저 업로드해주세요.');
            return;
        }

        const totalCount = uploadedImages.length;

        runBtn.textContent = '처리 중...';
        runBtn.disabled = true;

        initResults(totalCount);
        processedResults = [];

        showChartLabels = false;
        if (qualityChart) {
            qualityChart.data.datasets[0].data = [0, 0];
            qualityChart.update();
        }

        try {
            const formData = new FormData();

            uploadedImages.forEach(fileObj => {
                formData.append('images', fileObj.file);
            });

            formData.append('csv', uploadedCSV);

            const response = await fetch('/api/quality/run', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok && result.success) {
                processedResults = result.results;

                result.results.forEach((imageResult, index) => {
                    addResult(imageResult, index + 1, totalCount);
                });

                completeResults(result.results.length, totalCount);
            } else {
                throw new Error(result.message || '처리 중 오류가 발생했습니다.');
            }

        } catch (error) {
            console.error('오류:', error);
            alert('처리 중 오류가 발생했습니다: ' + error.message);
        } finally {
            runBtn.textContent = '실행';
            runBtn.disabled = false;
        }
    });

    let selectedSaveFormat = 'individual';

    const saveOptions = document.querySelectorAll('.save-option');
    saveOptions.forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedSaveFormat = option.dataset.format;
            executeSave();
        });
    });

    async function executeSave() {
        if (processedResults.length === 0) {
            alert('저장할 결과가 없습니다.');
            return;
        }

        try {
            if (selectedSaveFormat === 'zip') {
                if (typeof JSZip === 'undefined') {
                    const script = document.createElement('script');
                    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
                    await new Promise((resolve, reject) => {
                        script.onload = resolve;
                        script.onerror = reject;
                        document.head.appendChild(script);
                    });
                }

                const zip = new JSZip();

                for (const result of processedResults) {
                    const response = await fetch(result.quality_image);
                    const blob = await response.blob();
                    zip.file(`quality_${result.filename}`, blob);
                }

                const zipBlob = await zip.generateAsync({ type: 'blob' });

                const handle = await window.showSaveFilePicker({
                    suggestedName: 'quality_images.zip',
                    types: [{
                        description: 'ZIP Archive',
                        accept: { 'application/zip': ['.zip'] }
                    }]
                });

                const writable = await handle.createWritable();
                await writable.write(zipBlob);
                await writable.close();

            } else {
                for (const result of processedResults) {
                    const response = await fetch(result.quality_image);
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);

                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `quality_${result.filename}`;
                    a.style.display = 'none';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);

                    await new Promise(resolve => setTimeout(resolve, 300));
                }
            }

        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('저장 오류:', error);
                alert('저장 중 오류가 발생했습니다.');
            }
        }
    }
});