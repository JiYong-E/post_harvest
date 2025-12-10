document.addEventListener('DOMContentLoaded', () => {
    console.log('데이터 페이지 초기화 완료');

    let selectedDataType = 'mask'; // 기본값
    let currentData = [];

    const dataTypeButtons = document.querySelectorAll('.data-type-btn');
    dataTypeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            dataTypeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedDataType = btn.dataset.type;
            console.log('선택된 데이터 타입:', selectedDataType);
            loadData(selectedDataType);
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

    let selectedSaveFormat = 'csv';

    const saveOptions = document.querySelectorAll('.save-option');
    saveOptions.forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedSaveFormat = option.dataset.format;
            executeSave();
        });
    });

    async function loadData(dataType) {
        const dataContent = document.getElementById('dataContent');

        dataContent.innerHTML = `
            <div class="empty-data-state">
                <div class="empty-icon">
                    <img src="/static/images/icon_image.png" alt="로딩" width="80" height="80">
                </div>
                <h3>데이터를 불러오는 중...</h3>
            </div>
        `;

        try {
            const response = await fetch(`/api/data/${dataType}`);
            const result = await response.json();

            if (response.ok && result.success && result.data.length > 0) {
                currentData = result.data;
                displayData(result.data, dataType);
            } else {
                showEmptyState('데이터가 없습니다');
            }
        } catch (error) {
            console.error('데이터 로드 오류:', error);
            showEmptyState('데이터를 불러올 수 없습니다');
        }
    }

    function displayData(data, dataType) {
        const dataContent = document.getElementById('dataContent');

        if (!data || data.length === 0) {
            showEmptyState('데이터가 없습니다');
            return;
        }

        const headers = Object.keys(data[0]);

        let tableHTML = `
            <table class="data-table">
                <thead>
                    <tr>
                        ${headers.map(h => `<th>${h}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
        `;

        data.forEach(row => {
            tableHTML += '<tr>';
            headers.forEach(header => {
                tableHTML += `<td>${row[header] || '-'}</td>`;
            });
            tableHTML += '</tr>';
        });

        tableHTML += `
                </tbody>
            </table>
        `;

        dataContent.innerHTML = tableHTML;
    }

    function showEmptyState(message) {
        const dataContent = document.getElementById('dataContent');
        dataContent.innerHTML = `
            <div class="empty-data-state">
                <div class="empty-icon">
                    <img src="/static/images/icon_image.png" alt="데이터" width="80" height="80">
                </div>
                <h3>${message}</h3>
                <p class="sub-text">${selectedDataType === 'mask' ? '마스크' : selectedDataType === 'quality' ? '품질 분류' : '특징 추출'} 데이터</p>
            </div>
        `;
    }

    async function executeSave() {
        if (currentData.length === 0) {
            alert('저장할 데이터가 없습니다.');
            return;
        }

        try {
            if (selectedSaveFormat === 'csv') {
                downloadCSV(currentData);
            } else if (selectedSaveFormat === 'excel') {
                downloadExcel(currentData);
            }
        } catch (error) {
            console.error('저장 오류:', error);
            alert('저장 중 오류가 발생했습니다.');
        }
    }

    function downloadCSV(data) {
        const headers = Object.keys(data[0]);
        let csvContent = headers.join(',') + '\n';

        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header] || '';
                if (value.toString().includes(',') || value.toString().includes('\n')) {
                    return `"${value}"`;
                }
                return value;
            });
            csvContent += values.join(',') + '\n';
        });

        const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedDataType}_data.csv`;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    function downloadExcel(data) {
        const headers = Object.keys(data[0]);
        let excelContent = '<table><thead><tr>';

        headers.forEach(header => {
            excelContent += `<th>${header}</th>`;
        });
        excelContent += '</tr></thead><tbody>';

        data.forEach(row => {
            excelContent += '<tr>';
            headers.forEach(header => {
                excelContent += `<td>${row[header] || ''}</td>`;
            });
            excelContent += '</tr>';
        });

        excelContent += '</tbody></table>';

        const blob = new Blob([excelContent], { type: 'application/vnd.ms-excel' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedDataType}_data.xls`;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
    loadData('mask');
});