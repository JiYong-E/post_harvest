document.addEventListener('DOMContentLoaded', () => {
    console.log('특징 추출 시스템 초기화 완료');

    const fileList = document.getElementById('fileList');
    if (fileList) {
        fileList.style.overflowY = 'auto';
        fileList.style.maxHeight = 'calc(100vh - 350px)';
        fileList.style.minHeight = '200px';
    }

    let uploadedFiles = [];
    let processedResults = [];
    let selectedFruit = 'melon_browning';

    const fruitButtons = document.querySelectorAll('.fruit-btn');
    fruitButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            fruitButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedFruit = btn.dataset.fruit;
            console.log('선택된 특징:', selectedFruit);

            resetUploadArea();
        });
    });

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
    const fileInput = document.getElementById('fileInput');
    const fileSelectBtn = document.getElementById('fileSelectBtn');
    const emptyState = document.getElementById('emptyState');

    fileSelectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropArea.addEventListener('click', (e) => {
        if (e.target === dropArea || e.target.closest('.empty-state')) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
        fileInput.value = '';
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
        handleFiles(e.dataTransfer.files);
    });

    function handleFiles(files) {
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

            uploadedFiles.push(fileObj);
            createFileCard(fileObj);
        });

        updateEmptyState();
    }

    function createFileCard(fileObj) {
        const card = document.createElement('div');
        card.className = 'file-card';
        card.dataset.fileId = fileObj.id;

        const reader = new FileReader();
        reader.onload = (e) => {
            card.innerHTML = `
                <button class="delete-btn" onclick="deleteFile(${fileObj.id})">
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
        if (uploadedFiles.length > 0) {
            emptyState.classList.add('hidden');
        } else {
            emptyState.classList.remove('hidden');
        }
    }

    function resetUploadArea() {
        uploadedFiles = [];
        fileList.innerHTML = '';
        emptyState.classList.remove('hidden');
        console.log('이미지 업로드 영역 초기화됨');
    }

    window.deleteFile = function(fileId) {
        uploadedFiles = uploadedFiles.filter(f => f.id !== fileId);

        const card = document.querySelector(`[data-file-id="${fileId}"]`);
        if (card) {
            card.remove();
        }

        updateEmptyState();
        console.log('파일 삭제됨. 남은 파일:', uploadedFiles.length);
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
            resultImages.style.maxHeight = 'calc(100vh - 300px)';
            resultImages.style.minHeight = '200px';
        }
    }

    function addResult(result, currentIndex, totalCount, fruitType) {
        const resultImages = document.getElementById('resultImages');
        const progressText = document.getElementById('progressText');

        const resultItem = document.createElement('div');
        resultItem.className = 'result-image-item';

        if (result.detection_image && result.extraction_image) {
            let label1, label2;
            if (fruitType === 'melon_browning') {
                label1 = '갈변 탐지';
                label2 = '갈변 추출';
            } else if (fruitType === 'apple_lenticel') {
                label1 = '과점 탐지';
                label2 = '과점 추출';
            }

            resultItem.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <img src="${result.detection_image}" alt="${label1} ${currentIndex}" class="clickable-image" style="cursor: pointer;">
                        <p>${label1} - ${result.filename}</p>
                    </div>
                    <div>
                        <img src="${result.extraction_image}" alt="${label2} ${currentIndex}" class="clickable-image" style="cursor: pointer;">
                        <p>${label2} - ${result.filename}</p>
                    </div>
                </div>
            `;
        } else {
            resultItem.innerHTML = `
                <img src="${result.extracted_image}" alt="특징 추출 결과 ${currentIndex}" class="clickable-image" style="cursor: pointer;">
                <p>${result.filename}</p>
            `;
        }

        resultImages.appendChild(resultItem);

        const images = resultItem.querySelectorAll('.clickable-image');
        images.forEach(img => {
            img.addEventListener('click', () => openImageModal(img.src, img.alt));
        });

        progressText.textContent = `총 ${totalCount}개 이미지 처리중 (${currentIndex}/${totalCount})`;
    }

    function completeResults(successCount, totalCount) {
        const progressText = document.getElementById('progressText');
        if (successCount === 0) {
            progressText.innerHTML = `<span style="color: #ef4444;">과일 특징을 추출하지 못했습니다 (0/${totalCount})</span>`;
        } else {
            progressText.textContent = `총 ${totalCount}개 이미지 처리 완료 (${successCount}/${totalCount})`;
        }
    }

    function openImageModal(imageSrc, imageAlt) {
        const modal = document.createElement('div');
        modal.id = 'imageModal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            cursor: pointer;
        `;

        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            position: relative;
            max-width: 90%;
            max-height: 90%;
            display: flex;
            flex-direction: column;
            align-items: center;
        `;

        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            position: absolute;
            top: -40px;
            right: 0;
            background: white;
            border: none;
            font-size: 32px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            color: #333;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: all 0.2s;
        `;
        closeBtn.onmouseover = () => {
            closeBtn.style.background = '#f0f0f0';
            closeBtn.style.transform = 'scale(1.1)';
        };
        closeBtn.onmouseout = () => {
            closeBtn.style.background = 'white';
            closeBtn.style.transform = 'scale(1)';
        };

        const img = document.createElement('img');
        img.src = imageSrc;
        img.alt = imageAlt;
        img.style.cssText = `
            max-width: 100%;
            max-height: 85vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        `;

        const caption = document.createElement('div');
        caption.textContent = imageAlt;
        caption.style.cssText = `
            color: white;
            margin-top: 16px;
            font-size: 16px;
            text-align: center;
        `;

        img.onclick = (e) => e.stopPropagation();
        modalContent.onclick = (e) => e.stopPropagation();
        modalContent.appendChild(closeBtn);
        modalContent.appendChild(img);
        modalContent.appendChild(caption);
        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        const closeModal = () => {
            modal.remove();
        };

        modal.onclick = closeModal;
        closeBtn.onclick = closeModal;

        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
    }

    const runBtn = document.getElementById('runBtn');
    runBtn.addEventListener('click', async () => {
        if (uploadedFiles.length === 0) {
            alert('이미지를 먼저 업로드해주세요.');
            return;
        }

        const totalCount = uploadedFiles.length;

        runBtn.textContent = '처리 중...';
        runBtn.disabled = true;

        initResults(totalCount);
        processedResults = [];

        try {
            let successCount = 0;

            for (let i = 0; i < uploadedFiles.length; i++) {
                const fileObj = uploadedFiles[i];
                const formData = new FormData();
                formData.append('files', fileObj.file);
                formData.append('fruit_type', selectedFruit);

                try {
                    const response = await fetch('/api/extraction/run', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok && result.success && result.results.length > 0) {
                        const imageResult = result.results[0];
                        processedResults.push({
                            ...imageResult,
                            fruitType: selectedFruit
                        });
                        successCount++;

                        addResult(imageResult, successCount, totalCount, selectedFruit);
                    }
                } catch (error) {
                    console.error(`${fileObj.name} 처리 오류:`, error);
                }
            }

            completeResults(successCount, totalCount);

        } catch (error) {
            console.error('오류:', error);
            alert('처리 중 오류가 발생했습니다.');
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
                    if (result.detection_image && result.extraction_image) {
                        const response1 = await fetch(result.detection_image);
                        const blob1 = await response1.blob();

                        const response2 = await fetch(result.extraction_image);
                        const blob2 = await response2.blob();

                        if (result.fruitType === 'melon_browning') {
                            zip.file(`browning_detection_${result.filename}`, blob1);
                            zip.file(`browning_extraction_${result.filename}`, blob2);
                        } else if (result.fruitType === 'apple_lenticel') {
                            zip.file(`lenticel_visualization_${result.filename}`, blob1);
                            zip.file(`lenticel_masked_${result.filename}`, blob2);
                        }
                    } else {
                        const response = await fetch(result.extracted_image);
                        const blob = await response.blob();
                        zip.file(`extracted_${result.filename}`, blob);
                    }
                }

                const zipBlob = await zip.generateAsync({ type: 'blob' });

                const handle = await window.showSaveFilePicker({
                    suggestedName: 'extracted_images.zip',
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
                    if (result.detection_image && result.extraction_image) {
                        const response1 = await fetch(result.detection_image);
                        const blob1 = await response1.blob();
                        const url1 = window.URL.createObjectURL(blob1);
                        const a1 = document.createElement('a');
                        a1.href = url1;

                        if (result.fruitType === 'melon_browning') {
                            a1.download = `browning_detection_${result.filename}`;
                        } else if (result.fruitType === 'apple_lenticel') {
                            a1.download = `lenticel_visualization_${result.filename}`;
                        }

                        a1.style.display = 'none';
                        document.body.appendChild(a1);
                        a1.click();
                        document.body.removeChild(a1);
                        window.URL.revokeObjectURL(url1);

                        await new Promise(resolve => setTimeout(resolve, 300));

                        const response2 = await fetch(result.extraction_image);
                        const blob2 = await response2.blob();
                        const url2 = window.URL.createObjectURL(blob2);
                        const a2 = document.createElement('a');
                        a2.href = url2;

                        if (result.fruitType === 'melon_browning') {
                            a2.download = `browning_extraction_${result.filename}`;
                        } else if (result.fruitType === 'apple_lenticel') {
                            a2.download = `lenticel_masked_${result.filename}`;
                        }

                        a2.style.display = 'none';
                        document.body.appendChild(a2);
                        a2.click();
                        document.body.removeChild(a2);
                        window.URL.revokeObjectURL(url2);
                    } else {
                        const response = await fetch(result.extracted_image);
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `extracted_${result.filename}`;
                        a.style.display = 'none';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                    }

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