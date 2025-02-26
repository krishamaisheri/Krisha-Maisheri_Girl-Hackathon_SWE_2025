// patient_detail.js - Patient profile functionality
$(document).ready(function() {
    // Initialize confidence meters
    initializeConfidenceCircles();
    
    // Load analyses if they're not already loaded
    if ($('.analysis-timeline').length > 0 && $('.timeline-item').length === 0) {
        loadPatientAnalyses();
    }
});

function loadPatientAnalyses() {
    // Get patient ID from URL
    const patientId = window.location.pathname.split('/').pop();
    
    API.get(`/api/patients/${patientId}/analyses`)
        .then(analyses => {
            const container = $('.analysis-timeline');
            container.empty();
            
            if (analyses.length === 0) {
                container.html(`
                    <div class="no-analyses">
                        <div class="no-data-icon">
                            <i class="fas fa-file-medical-alt"></i>
                        </div>
                        <p>No analyses have been performed yet.</p>
                        <a href="/new_analysis/${patientId}" class="btn btn-primary">
                            <i class="fas fa-plus"></i> Perform New Analysis
                        </a>
                    </div>
                `);
                return;
            }
            
            analyses.forEach(analysis => {
                let iconClass = 'fas fa-file-medical-alt';
                if (analysis.image_type === 'brain') {
                    iconClass = 'fas fa-brain';
                } else if (analysis.image_type === 'chest') {
                    iconClass = 'fas fa-lungs';
                }
                
                const analysisType = analysis.image_type ? 
                    analysis.image_type.charAt(0).toUpperCase() + analysis.image_type.slice(1) : 
                    'Medical';
                
                const timelineItem = `
                    <div class="timeline-item">
                        <div class="timeline-marker">
                            <i class="${iconClass}"></i>
                        </div>
                        <div class="timeline-content">
                            <h4>${analysisType} Analysis</h4>
                            <div class="timestamp">${formatDateTime(analysis.created_at)}</div>
                            
                            <div class="analysis-details">
                                ${analysis.prediction ? `
                                <div class="prediction-result">
                                    <div class="prediction-label">
                                        <span>Prediction:</span> ${analysis.prediction}
                                    </div>
                                    
                                    <div class="confidence-meter">
                                        <div class="confidence-circle" data-value="${analysis.confidence}">
                                            <span class="confidence-text">${analysis.confidence.toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>
                                ` : ''}
                                
                                <div class="analysis-actions">
                                    <a href="/analysis/${analysis.id}" class="btn btn-info btn-sm">
                                        <i class="fas fa-eye"></i> View Details
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                container.append(timelineItem);
            });
            
            // Re-initialize confidence circles after adding analyses
            initializeConfidenceCircles();
        })
        .catch(error => {
            console.error('Error loading analyses:', error);
            $('.analysis-timeline').html(`
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error loading analyses. Please refresh the page to try again.</p>
                </div>
            `);
        });
}