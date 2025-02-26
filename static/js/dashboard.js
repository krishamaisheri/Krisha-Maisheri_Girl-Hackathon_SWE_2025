// dashboard.js - Dashboard functionality
$(document).ready(function() {
    // Load dashboard data
    loadDashboardData();
    
    // Set up event listeners
    $("#refresh-dashboard").on("click", function() {
        $(this).find('i').addClass('fa-spin');
        loadDashboardData().then(() => {
            setTimeout(() => {
                $(this).find('i').removeClass('fa-spin');
            }, 500);
        });
    });
    
    $("#export-data").on("click", function() {
        exportDashboardData();
    });
    
    // Initialize charts
    initializeCharts();
});

// Load all dashboard data
function loadDashboardData() {
    return Promise.all([
        loadStatistics(),
        loadRecentPatients(),
        updateCharts()
    ]);
}

// Load statistics counters
function loadStatistics() {
    return API.get('/api/statistics')
        .then(data => {
            $("#patient-count").text(data.patients || 0);
            $("#analysis-count").text(data.analyses || 0);
            $("#brain-count").text(data.brainScans || 0);
            $("#chest-count").text(data.chestScans || 0);
        })
        .catch(error => {
            console.error('Error loading statistics:', error);
            // For demo, generate random numbers
            $("#patient-count").text(Math.floor(Math.random() * 500) + 100);
            $("#analysis-count").text(Math.floor(Math.random() * 1000) + 200);
            $("#brain-count").text(Math.floor(Math.random() * 500) + 100);
            $("#chest-count").text(Math.floor(Math.random() * 500) + 100);
        });
}

// Load recent patients
function loadRecentPatients() {
    return API.get('/api/patients?limit=5')
        .then(patients => {
            const tbody = $("#recent-patients-list");
            tbody.empty();
            
            if (patients.length === 0) {
                tbody.append(`
                    <tr>
                        <td colspan="4" class="text-center">No patients found</td>
                    </tr>
                `);
                return;
            }
            
            patients.forEach(patient => {
                tbody.append(`
                    <tr>
                        <td>${patient.first_name} ${patient.last_name}</td>
                        <td>${patient.gender}</td>
                        <td>${formatDate(patient.date_of_birth)}</td>
                        <td>
                            <a href="/patient/${patient.id}" class="btn btn-sm btn-info">
                                <i class="fas fa-eye"></i>
                            </a>
                        </td>
                    </tr>
                `);
            });
        })
        .catch(error => {
            console.error('Error loading recent patients:', error);
            // For demo, show placeholder
            const tbody = $("#recent-patients-list");
            tbody.html(`
                <tr>
                    <td>John Doe</td>
                    <td>Male</td>
                    <td>1985-06-12</td>
                    <td>
                        <a href="#" class="btn btn-sm btn-info">
                            <i class="fas fa-eye"></i>
                        </a>
                    </td>
                </tr>
                <tr>
                    <td>Jane Smith</td>
                    <td>Female</td>
                    <td>1990-03-24</td>
                    <td>
                        <a href="#" class="btn btn-sm btn-info">
                            <i class="fas fa-eye"></i>
                        </a>
                    </td>
                </tr>
            `);
        });
}

// Initialize charts
function initializeCharts() {
    // Analysis distribution chart
    const analysisCtx = document.getElementById('analysisChart').getContext('2d');
    window.analysisChart = new Chart(analysisCtx, {
        type: 'doughnut',
        data: {
            labels: ['Brain Scans', 'Chest Scans', 'Other'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    '#4e73df',
                    '#1cc88a',
                    '#36b9cc'
                ],
                hoverBackgroundColor: [
                    '#2e59d9',
                    '#17a673',
                    '#2c9faf'
                ],
                borderWidth: 1
            }]
        },
        options: {
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            cutout: '70%'
        }
    });
    
    // Confidence chart
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    window.confidenceChart = new Chart(confidenceCtx, {
        type: 'bar',
        data: {
            labels: ['90-100%', '80-89%', '70-79%', '60-69%', 'Below 60%'],
            datasets: [{
                label: 'Prediction Confidence',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(23, 162, 184, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(220, 53, 69, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(23, 162, 184)',
                    'rgb(255, 193, 7)',
                    'rgb(255, 159, 64)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Analyses'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Confidence Range'
                    }
                }
            }
        }
    });
    
    // Update charts with data
    updateCharts();
}

// Update chart data
function updateCharts() {
    return API.get('/api/statistics/charts')
        .catch(error => {
            console.error('Error loading chart data:', error);
            // For demo, return placeholder data
            return {
                analysisDistribution: {
                    brainScans: Math.floor(Math.random() * 100) + 50,
                    chestScans: Math.floor(Math.random() * 100) + 50,
                    other: Math.floor(Math.random() * 20)
                },
                confidenceDistribution: {
                    '90-100': Math.floor(Math.random() * 40) + 30,
                    '80-89': Math.floor(Math.random() * 30) + 20,
                    '70-79': Math.floor(Math.random() * 20) + 15,
                    '60-69': Math.floor(Math.random() * 15) + 5,
                    'below60': Math.floor(Math.random() * 10)
                }
            };
        })
        .then(data => {
            // Update analysis distribution chart
            if (window.analysisChart) {
                window.analysisChart.data.datasets[0].data = [
                    data.analysisDistribution.brainScans || 0,
                    data.analysisDistribution.chestScans || 0,
                    data.analysisDistribution.other || 0
                ];
                window.analysisChart.update();
            }
            
            // Update confidence chart
            if (window.confidenceChart) {
                window.confidenceChart.data.datasets[0].data = [
                    data.confidenceDistribution['90-100'] || 0,
                    data.confidenceDistribution['80-89'] || 0,
                    data.confidenceDistribution['70-79'] || 0,
                    data.confidenceDistribution['60-69'] || 0,
                    data.confidenceDistribution.below60 || 0
                ];
                window.confidenceChart.update();
            }
        });
}

// Export dashboard data
function exportDashboardData() {
    alert('This would download a CSV or Excel file with dashboard data in a real application.');
}