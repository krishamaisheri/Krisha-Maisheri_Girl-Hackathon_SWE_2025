// diagnosis.js - Symptom diagnosis system frontend functionality
$(document).ready(function() {
    // Load patient diagnosis history on page load
    loadDiagnosisHistory();
    
    // Handle form submission
    $('#diagnosisForm').on('submit', function(e) {
        e.preventDefault();
        submitDiagnosis();
    });
    
    // Toggle new disease form
    $('#newDiseaseCheck').on('change', function() {
        $('#newDiseaseForm').toggleClass('hidden', !this.checked);
    });
    
    // Add precaution field
    $('#addPrecaution').on('click', function() {
        addPrecautionField();
    });
    
    // Remove precaution field (delegated event for dynamically created elements)
    $('#precautions-list').on('click', '.remove-precaution', function() {
        $(this).closest('.input-group').remove();
    });
    
    // Select disease card
    $(document).on('click', '.disease-card', function() {
        $('.disease-card').removeClass('selected');
        $(this).addClass('selected');
    });
    
    // Confirm diagnosis
    $('#confirmDiagnosis').on('click', function() {
        confirmDiagnosis();
    });
});

// Submit symptoms for diagnosis
function submitDiagnosis() {
    const patientId = $('#patientId').val();
    const symptoms = $('#symptoms').val().trim();
    
    if (!symptoms) {
        showAlert('Please enter symptoms', 'danger');
        return;
    }
    
    // Show loading indicator
    $('#loadingIndicator').removeClass('hidden');
    $('#diagnosisResults').addClass('hidden');
    
    // Send diagnosis request
    $.ajax({
        url: '/diagnosis/suggest',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            patient_id: patientId,
            symptoms: symptoms
        }),
        success: function(response) {
            // Hide loading indicator
            $('#loadingIndicator').addClass('hidden');
            
            // Store diagnosis ID for confirmation
            sessionStorage.setItem('currentDiagnosisId', response.diagnosis_id);
            
            // Display suggestions
            displaySuggestions(response.suggestions);
            
            // Display AI notes
            $('#ai-notes').html(response.ai_notes);
            
            // Show results
            $('#diagnosisResults').removeClass('hidden');
        },
        error: function(xhr) {
            // Hide loading indicator
            $('#loadingIndicator').addClass('hidden');
            
            let errorMessage = 'Failed to process diagnosis';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMessage = xhr.responseJSON.error;
            }
            
            showAlert(errorMessage, 'danger');
        }
    });
}

// Display disease suggestions
function displaySuggestions(suggestions) {
    const container = $('#suggestions-container');
    container.empty();
    
    if (!suggestions || suggestions.length === 0) {
        container.html('<div class="alert alert-info">No specific diseases match these symptoms.</div>');
        return;
    }
    
    // Sort by confidence (highest first)
    suggestions.sort((a, b) => b.confidence - a.confidence);
    
    suggestions.forEach((suggestion, index) => {
        const confidencePercent = (suggestion.confidence * 100).toFixed(1);
        const card = $(`
            <div class="disease-card ${index === 0 ? 'selected' : ''}" data-disease-id="${suggestion.disease_id}">
                <div class="disease-header">
                    <div class="disease-name">${suggestion.name}</div>
                    <div class="confidence-badge">${confidencePercent}% match</div>
                </div>
                <div class="disease-precautions">
                    ${suggestion.precautions && suggestion.precautions.length > 0 ? 
                        `<strong>Recommended precautions:</strong>
                        <ul class="precautions-list">
                            ${suggestion.precautions.map(p => `<li>${p}</li>`).join('')}
                        </ul>` : 
                        '<p>No specific precautions available for this condition.</p>'}
                </div>
            </div>
        `);
        
        container.append(card);
    });
}

// Add new precaution field
function addPrecautionField() {
    const precautionCount = $('.precaution-input').length + 1;
    const newField = $(`
        <div class="input-group mb-2">
            <input type="text" class="form-control precaution-input" placeholder="Precaution ${precautionCount}">
            <div class="input-group-append">
                <button class="btn btn-outline-secondary remove-precaution" type="button">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `);
    
    $('#precautions-list').append(newField);
}

// Confirm diagnosis
function confirmDiagnosis() {
    const diagnosisId = sessionStorage.getItem('currentDiagnosisId');
    if (!diagnosisId) {
        showAlert('No active diagnosis to confirm', 'danger');
        return;
    }
    
    const doctorNotes = $('#doctorNotes').val();
    const isNewDisease = $('#newDiseaseCheck').is(':checked');
    let selectedDiseaseId = $('.disease-card.selected').data('disease-id');
    
    const data = {
        doctor_notes: doctorNotes
    };
    
    // If adding new disease
    if (isNewDisease) {
        const newDiseaseName = $('#newDiseaseName').val().trim();
        if (!newDiseaseName) {
            showAlert('Please enter a name for the new disease', 'danger');
            return;
        }
        
        // Collect precautions
        const precautions = [];
        $('.precaution-input').each(function() {
            const value = $(this).val().trim();
            if (value) {
                precautions.push(value);
            }
        });
        
        data.new_disease = {
            name: newDiseaseName,
            precautions: precautions
        };
    }
    
    // Show loading
    $('#confirmDiagnosis').prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');
    
    // Send confirmation request
    $.ajax({
        url: `/diagnosis/confirm/${diagnosisId}`,
        type: 'PUT',
        contentType: 'application/json',
        data: JSON.stringify(data),
        success: function(response) {
            showAlert('Diagnosis confirmed successfully', 'success');
            
            // Clear form and results
            $('#symptoms').val('');
            $('#doctorNotes').val('');
            $('#newDiseaseCheck').prop('checked', false);
            $('#newDiseaseForm').addClass('hidden');
            $('#newDiseaseName').val('');
            $('#precautions-list').html('');
            $('#diagnosisResults').addClass('hidden');
            
            // Clear stored diagnosis ID
            sessionStorage.removeItem('currentDiagnosisId');
            
            // Reload diagnosis history
            loadDiagnosisHistory();
            
            // Re-enable button
            $('#confirmDiagnosis').prop('disabled', false).html('<i class="fas fa-check"></i> Confirm Diagnosis');
        },
        error: function(xhr) {
            let errorMessage = 'Failed to confirm diagnosis';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMessage = xhr.responseJSON.error;
            }
            
            showAlert(errorMessage, 'danger');
            
            // Re-enable button
            $('#confirmDiagnosis').prop('disabled', false).html('<i class="fas fa-check"></i> Confirm Diagnosis');
        }
    });
}

// Load patient diagnosis history
function loadDiagnosisHistory() {
    const patientId = $('#patientId').val();
    
    $.ajax({
        url: `/diagnosis/history/${patientId}`,
        type: 'GET',
        success: function(response) {
            displayDiagnosisHistory(response);
        },
        error: function() {
            $('#history-container').html('<div class="alert alert-danger">Failed to load diagnosis history.</div>');
        }
    });
}

// Display diagnosis history
function displayDiagnosisHistory(diagnoses) {
    const container = $('#history-container');
    container.empty();
    
    if (!diagnoses || diagnoses.length === 0) {
        container.html('<p class="text-center">No previous diagnoses found for this patient.</p>');
        return;
    }
    
    diagnoses.forEach(diagnosis => {
        const historyItem = $(`
            <div class="history-item">
                <div class="history-header">
                    <span class="history-date">
                        <i class="fas fa-calendar-alt"></i> ${diagnosis.created_at}
                    </span>
                    <span class="status-badge ${diagnosis.is_approved ? 'status-approved' : 'status-pending'}">
                        ${diagnosis.is_approved ? 'Confirmed' : 'Pending'}
                    </span>
                </div>
                <div class="history-disease">
                    <i class="fas fa-clipboard-check"></i> 
                    ${diagnosis.disease ? diagnosis.disease : 'No specific diagnosis'}
                </div>
                <div class="history-symptoms">
                    <strong>Symptoms:</strong> ${diagnosis.symptoms}
                </div>
                ${diagnosis.doctor_notes ? `
                <div class="history-notes">
                    <strong>Notes:</strong>
                    <div class="notes-content">${diagnosis.doctor_notes}</div>
                </div>
                ` : ''}
            </div>
        `);
        
        container.append(historyItem);
    });
}

// Show alert message
function showAlert(message, type) {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                <span aria-hidden="true">&times;</span>
            </button>
        </div>
    `;
    
    // Remove existing alerts
    $('.alert-dismissible').alert('close');
    
    // Add new alert before the form
    $('.symptom-form').before(alertHtml);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        $('.alert-dismissible').alert('close');
    }, 5000);
}