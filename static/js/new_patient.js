// new_patient.js - Add new patient functionality
$(document).ready(function() {
    // Initialize the new patient form
    initNewPatientForm();
});

function initNewPatientForm() {
    // Set max date for date of birth to today
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');
    const todayString = `${yyyy}-${mm}-${dd}`;
    $("#date_of_birth").attr('max', todayString);
    
    // Form validation and submission
    $("#new-patient-form").on("submit", function(e) {
        e.preventDefault();
        
        // Basic form validation
        if (!validatePatientForm()) {
            return false;
        }
        
        // Create FormData object
        const formData = new FormData(this);
        
        // Submit the form
        submitPatientForm(formData);
    });
}

function validatePatientForm() {
    let isValid = true;
    const requiredFields = ['first_name', 'last_name', 'date_of_birth', 'gender'];
    
    // Check required fields
    requiredFields.forEach(field => {
        const value = $(`#${field}`).val();
        if (!value || value.trim() === '') {
            isValid = false;
            $(`#${field}`).addClass('error');
        } else {
            $(`#${field}`).removeClass('error');
        }
    });
    
    // Validate email if provided
    const email = $("#email").val();
    if (email && !isValidEmail(email)) {
        isValid = false;
        $("#email").addClass('error');
    } else {
        $("#email").removeClass('error');
    }
    
    // Display error message if form is invalid
    if (!isValid) {
        alert('Please fill in all required fields correctly');
    }
    
    return isValid;
}

function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function submitPatientForm(formData) {
    // Show loading indicator
    const submitBtn = $("#new-patient-form button[type='submit']");
    const originalText = submitBtn.html();
    submitBtn.html('<i class="fas fa-spinner fa-spin"></i> Saving...');
    submitBtn.prop('disabled', true);
    
    // Submit form via API
    fetch('/api/patients', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            // Show success message
            alert('Patient added successfully!');
            
            // Redirect to patient detail page
            window.location.href = `/patient/${data.patient.id}`;
        } else {
            // Show error message
            alert(`Error: ${data.message}`);
            submitBtn.html(originalText);
            submitBtn.prop('disabled', false);
        }
    })
    .catch(error => {
        console.error('Error submitting form:', error);
        alert('An error occurred while saving the patient. Please try again.');
        submitBtn.html(originalText);
        submitBtn.prop('disabled', false);
    });
}