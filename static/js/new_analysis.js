// new_analysis.js - New analysis functionality
$(document).ready(function() {
    // Initialize file uploads
    initFileUploads();
    
    // Initialize form submission
    $("#analysis-form").on("submit", function(e) {
        e.preventDefault();
        submitAnalysisForm();
    });
});

function initFileUploads() {
    // Image upload preview
    $("#image_file").on("change", function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                $("#image-preview").html(`
                    <div class="preview-file">
                        <img src="${e.target.result}" alt="Image Preview">
                        <span class="file-name">${file.name}</span>
                        <button type="button" class="remove-file" data-target="image_file">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `);
                $("#image-upload").addClass("has-file");
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Report upload preview
    $("#report_file").on("change", function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                let previewContent;
                if (file.type === "application/pdf") {
                    previewContent = `
                        <div class="pdf-icon">
                            <i class="fas fa-file-pdf"></i>
                        </div>
                    `;
                } else {
                    previewContent = `<img src="${e.target.result}" alt="Report Preview">`;
                }
                
                $("#report-preview").html(`
                    <div class="preview-file">
                        ${previewContent}
                        <span class="file-name">${file.name}</span>
                        <button type="button" class="remove-file" data-target="report_file">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `);
                $("#report-upload").addClass("has-file");
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Remove file functionality
    $(document).on("click", ".remove-file", function() {
        const targetId = $(this).data("target");
        $(`#${targetId}`).val("");
        
        if (targetId === "image_file") {
            $("#image-preview").empty();
            $("#image-upload").removeClass("has-file");
        } else if (targetId === "report_file") {
            $("#report-preview").empty();
            $("#report-upload").removeClass("has-file");
        }
    });
    
    // Drag and drop functionality
    $(".file-upload-container").each(function() {
        const container = $(this);
        const input = container.find("input[type='file']");
        
        container.on("dragover", function(e) {
            e.preventDefault();
            container.addClass("drag-over");
        });
        
        container.on("dragleave", function(e) {
            e.preventDefault();
            container.removeClass("drag-over");
        });
        
        container.on("drop", function(e) {
            e.preventDefault();
            container.removeClass("drag-over");
            
            const files = e.originalEvent.dataTransfer.files;
            if (files.length > 0) {
                input[0].files = files;
                input.trigger("change");
            }
        });
        
        container.find(".upload-placeholder").on("click", function() {
            input.click();
        });
    });
}

function submitAnalysisForm() {
    // Validate form
    const imageType = $("#image_type").val();
    const imageFile = $("#image_file")[0].files[0];
    
    if (!imageType) {
        alert("Please select an analysis type");
        return;
    }
    
    if (!imageFile) {
        alert("Please upload a medical image for analysis");
        return;
    }
    
    // Show loading state
    $("#analyze-btn").prop("disabled", true);
    $("#analyze-btn .spinner").show();
    $("#analysis-results").show();
    $(".analysis-loading").show();
    $(".analysis-content").hide();
    
    // Prepare form data
    const formData = new FormData($("#analysis-form")[0]);
    
    // Get patient ID from hidden input
    const patientId = formData.get("patient_id");
    
    // Submit form
    fetch(`/api/patients/${patientId}/analyses`, {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();
    })
    .then(data => {
        if (data.status === "success") {
            displayAnalysisResults(data.analysis);
        } else {
            throw new Error(data.message || "Unknown error");
        }
    })
    .catch(error => {
        console.error("Error submitting analysis:", error);
        alert("Error: " + error.message);
        
        // Reset UI
        $("#analyze-btn").prop("disabled", false);
        $("#analyze-btn .spinner").hide();
        $("#analysis-results").hide();
    });
}

function displayAnalysisResults(analysis) {
    // Hide loading and show content
    $(".analysis-loading").hide();
    $(".analysis-content").show();
    
    // Update results
    $("#prediction-value").text(analysis.prediction || "N/A");
    
    // Set confidence meter
    const confidenceValue = analysis.confidence || 0;
    $("#confidence-value").text(confidenceValue.toFixed(1) + "%");
    
    // Update confidence meter visual
    const confidenceMeter = $("#confidence-meter");
    confidenceMeter.css("--progress", confidenceValue);
    
    // Set confidence color
    let confidenceColor;
    if (confidenceValue >= 90) {
        confidenceColor = "#28a745"; // Green
    } else if (confidenceValue >= 70) {
        confidenceColor = "#17a2b8"; // Blue
    } else if (confidenceValue >= 50) {
        confidenceColor = "#ffc107"; // Yellow
    } else {
        confidenceColor = "#dc3545"; // Red
    }
    confidenceMeter.css("--confidence-color", confidenceColor);
    
    // Set analysis report
    $("#analysis-report").html(analysis.analysis_text.replace(/\n/g, "<br>"));
    
    // Update view details link
    $("#view-details-btn").attr("href", `/analysis/${analysis.id}`);
    
    // Enable print functionality
    $("#print-results").on("click", function() {
        window.print();
    });
}