// patients.js - Patient list functionality
$(document).ready(function() {
    // Set up search functionality
    $("#patient-search").on("keyup", function() {
        const searchText = $(this).val().toLowerCase();
        $("#patients-list tr").filter(function() {
            $(this).toggle($(this).text().toLowerCase().indexOf(searchText) > -1);
        });
    });
    
    // Initialize patient list if empty
    if ($("#patients-list tr").length === 0) {
        loadPatientList();
    }
});

// Load patient list via API
function loadPatientList() {
    API.get('/api/patients')
        .then(patients => {
            const tbody = $("#patients-list");
            tbody.empty();
            
            if (patients.length === 0) {
                tbody.append(`
                    <tr>
                        <td colspan="6" class="text-center">No patients found</td>
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
                        <td>${patient.email || patient.phone || 'N/A'}</td>
                        <td>${formatDate(patient.created_at)}</td>
                        <td>
                            <div class="action-buttons">
                                <a href="/patient/${patient.id}" class="btn btn-sm btn-info" title="View Profile">
                                    <i class="fas fa-eye"></i>
                                </a>
                                <a href="/new_analysis/${patient.id}" class="btn btn-sm btn-primary" title="New Analysis">
                                    <i class="fas fa-file-medical"></i>
                                </a>
                            </div>
                        </td>
                    </tr>
                `);
            });
        })
        .catch(error => {
            console.error('Error loading patients:', error);
            $("#patients-list").html(`
                <tr>
                    <td colspan="6" class="text-center text-danger">
                        Error loading patient data. Please try again later.
                    </td>
                </tr>
            `);
        });
}